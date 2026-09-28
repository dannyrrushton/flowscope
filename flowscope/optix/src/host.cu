// Host side of the OptiX renderer, exposed as a small C API for ctypes
// (see flowscope/optix_backend.py). One renderer = one CUDA stream, one
// pipeline, a triangle GAS (the parts) + a round-linear curve GAS (the board
// traces) under one IAS, and the OptiX AI denoiser.
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>  // needs the stubs declared first

#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "shared.h"

using namespace fs;

namespace
{

#define FS_CUDA(call)                                                                               \
    do                                                                                              \
    {                                                                                               \
        cudaError_t e_ = (call);                                                                    \
        if (e_ != cudaSuccess)                                                                      \
            throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(e_));           \
    } while (0)

#define FS_OPTIX(call)                                                                              \
    do                                                                                              \
    {                                                                                               \
        OptixResult r_ = (call);                                                                    \
        if (r_ != OPTIX_SUCCESS)                                                                    \
            throw std::runtime_error(std::string(#call) + ": " + optixGetErrorName(r_) + " (" +     \
                                     optixGetErrorString(r_) + ")");                                \
    } while (0)

struct Buffer
{
    void* ptr = nullptr;
    size_t size = 0;
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { cudaFree(ptr); }
    void reserve(size_t bytes)
    {
        if (bytes <= size)
            return;
        cudaFree(ptr);
        ptr = nullptr;
        size = 0;
        FS_CUDA(cudaMalloc(&ptr, bytes));
        size = bytes;
    }
    void upload(const void* data, size_t bytes, cudaStream_t stream)
    {
        reserve(bytes ? bytes : 16);
        if (bytes)
            FS_CUDA(cudaMemcpyAsync(ptr, data, bytes, cudaMemcpyHostToDevice, stream));
    }
    CUdeviceptr d() const { return reinterpret_cast<CUdeviceptr>(ptr); }
};

template <typename T>
struct Record
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};
struct Empty
{
};

std::vector<char> read_file(const char* path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error(std::string("cannot open ") + path);
    return std::vector<char>(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

void log_cb(unsigned int level, const char* tag, const char* msg, void*)
{
    if (level <= 2)
        fprintf(stderr, "[optix][%s] %s\n", tag, msg);
}

void set_error(const std::exception& e, char* err, int errlen)
{
    if (err && errlen > 0)
    {
        strncpy(err, e.what(), errlen - 1);
        err[errlen - 1] = 0;
    }
}

// Bloom: bright parts of the HDR frame at half resolution, blurred, then added
// back in the tone mapper so hot traces and chips bleed a soft halo.
constexpr float BLOOM_THRESHOLD = 0.8f;
constexpr float BLOOM_STRENGTH = 0.55f;
constexpr int BLOOM_RADIUS = 12;

__global__ void bright_pass(const float4* in, float4* out, unsigned int w, unsigned int h, unsigned int bw,
                            unsigned int bh)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= bw || y >= bh)
        return;
    float r = 0, g = 0, b = 0;
    for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx)
        {
            float4 c = in[min(y * 2 + dy, h - 1) * w + min(x * 2 + dx, w - 1)];
            r += c.x;
            g += c.y;
            b += c.z;
        }
    r *= 0.25f;
    g *= 0.25f;
    b *= 0.25f;
    float lum = fmaxf(r, fmaxf(g, b));
    float k = lum > BLOOM_THRESHOLD ? (lum - BLOOM_THRESHOLD) / lum : 0.0f;
    out[y * bw + x] = make_float4(r * k, g * k, b * k, 0.0f);
}

__global__ void blur(const float4* in, float4* out, unsigned int bw, unsigned int bh, int dx, int dy)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= static_cast<int>(bw) || y >= static_cast<int>(bh))
        return;
    const float sigma = BLOOM_RADIUS * 0.5f;
    float r = 0, g = 0, b = 0, wsum = 0;
    for (int i = -BLOOM_RADIUS; i <= BLOOM_RADIUS; ++i)
    {
        int sx = min(max(x + i * dx, 0), static_cast<int>(bw) - 1);
        int sy = min(max(y + i * dy, 0), static_cast<int>(bh) - 1);
        float wgt = __expf(-0.5f * i * i / (sigma * sigma));
        float4 c = in[sy * bw + sx];
        r += c.x * wgt;
        g += c.y * wgt;
        b += c.z * wgt;
        wsum += wgt;
    }
    out[y * bw + x] = make_float4(r / wsum, g / wsum, b / wsum, 0.0f);
}

// ACES-ish filmic curve, sRGB encode, vignette → Cairo RGB24 (BGRX in memory).
__global__ void tonemap(const float4* in, const float4* bloom, uchar4* out, unsigned int w, unsigned int h,
                        unsigned int bw, unsigned int bh, float exposure)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    float4 c = in[y * w + x];
    {  // bilinear upsample of the half-resolution bloom
        float fx = fminf(fmaxf((x + 0.5f) * 0.5f - 0.5f, 0.0f), bw - 1.0f);
        float fy = fminf(fmaxf((y + 0.5f) * 0.5f - 0.5f, 0.0f), bh - 1.0f);
        unsigned int x0 = static_cast<unsigned int>(fx), y0 = static_cast<unsigned int>(fy);
        unsigned int x1 = min(x0 + 1, bw - 1), y1 = min(y0 + 1, bh - 1);
        float tx = fx - x0, ty = fy - y0;
        float4 a = bloom[y0 * bw + x0], b = bloom[y0 * bw + x1];
        float4 d = bloom[y1 * bw + x0], e = bloom[y1 * bw + x1];
        c.x += BLOOM_STRENGTH * ((a.x * (1 - tx) + b.x * tx) * (1 - ty) + (d.x * (1 - tx) + e.x * tx) * ty);
        c.y += BLOOM_STRENGTH * ((a.y * (1 - tx) + b.y * tx) * (1 - ty) + (d.y * (1 - tx) + e.y * tx) * ty);
        c.z += BLOOM_STRENGTH * ((a.z * (1 - tx) + b.z * tx) * (1 - ty) + (d.z * (1 - tx) + e.z * tx) * ty);
    }
    float fx = (x + 0.5f) / w - 0.5f, fy = (y + 0.5f) / h - 0.5f;
    float vig = 1.0f - 0.35f * (fx * fx + fy * fy) * 2.0f;
    float rgb[3] = {c.x, c.y, c.z};
    unsigned char o[3];
    for (int i = 0; i < 3; ++i)
    {
        float v = fmaxf(rgb[i], 0.0f) * exposure * vig;
        v = (v * (2.51f * v + 0.03f)) / (v * (2.43f * v + 0.59f) + 0.14f);
        v = fminf(fmaxf(v, 0.0f), 1.0f);
        v = v <= 0.0031308f ? 12.92f * v : 1.055f * powf(v, 1.0f / 2.4f) - 0.055f;
        o[i] = static_cast<unsigned char>(v * 255.0f + 0.5f);
    }
    out[y * w + x] = make_uchar4(o[2], o[1], o[0], 255);
}

}  // namespace

struct FsRenderer
{
    std::string device_name;
    cudaStream_t stream = nullptr;
    OptixDeviceContext context = nullptr;
    OptixModule module = nullptr, curve_module = nullptr;
    OptixPipeline pipeline = nullptr;
    std::vector<OptixProgramGroup> groups;
    OptixShaderBindingTable sbt = {};
    Buffer raygen_rec, miss_rec, hit_rec;

    Buffer vertices, tri_material, materials, curve_points, curve_widths, curve_index, segments;
    Buffer tri_gas, curve_gas, ias, instances, temp;
    OptixTraversableHandle tri_handle = 0, curve_handle = 0, top = 0;
    unsigned int num_materials = 0;

    Buffer pipes, color, albedo, normal, denoised, out, params_buf, bloom_a, bloom_b;
    unsigned int num_pipes = 0;  // set by fs_set_pipes
    Buffer den_state, den_scratch, den_intensity, den_avg;
    OptixDenoiser denoiser = nullptr;
    size_t den_scratch_size = 0;
    unsigned int width = 0, height = 0;

    static constexpr unsigned int BUILD_FLAGS =
        OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;

    explicit FsRenderer(const char* module_path)
    {
        FS_CUDA(cudaSetDevice(0));
        cudaDeviceProp prop;
        FS_CUDA(cudaGetDeviceProperties(&prop, 0));
        device_name = prop.name;
        FS_CUDA(cudaFree(nullptr));
        FS_CUDA(cudaStreamCreate(&stream));
        FS_OPTIX(optixInit());
        OptixDeviceContextOptions opts = {};
        opts.logCallbackFunction = &log_cb;
        opts.logCallbackLevel = 2;
        FS_OPTIX(optixDeviceContextCreate(nullptr, &opts, &context));
        build_pipeline(read_file(module_path));
        OptixDenoiserOptions dopts = {};
        dopts.guideAlbedo = 1;
        dopts.guideNormal = 1;
        FS_OPTIX(optixDenoiserCreate(context, OPTIX_DENOISER_MODEL_KIND_AOV, &dopts, &denoiser));
    }

    ~FsRenderer()
    {
        if (stream)
            cudaStreamSynchronize(stream);
        if (denoiser)
            optixDenoiserDestroy(denoiser);
        if (pipeline)
            optixPipelineDestroy(pipeline);
        for (auto g : groups)
            optixProgramGroupDestroy(g);
        if (module)
            optixModuleDestroy(module);
        if (context)
            optixDeviceContextDestroy(context);
        if (stream)
            cudaStreamDestroy(stream);
    }

    void build_pipeline(const std::vector<char>& code)
    {
        OptixModuleCompileOptions mopts = {};
        mopts.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        mopts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
        OptixPipelineCompileOptions popts = {};
        popts.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        popts.numPayloadValues = 2;
        popts.numAttributeValues = 2;
        popts.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        popts.pipelineLaunchParamsVariableName = "params";
        popts.usesPrimitiveTypeFlags =
            OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR;

        char log[4096];
        size_t log_size = sizeof(log);
        FS_OPTIX(optixModuleCreate(context, &mopts, &popts, code.data(), code.size(), log, &log_size, &module));
        OptixBuiltinISOptions is = {};
        is.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR;
        is.buildFlags = BUILD_FLAGS;
        FS_OPTIX(optixBuiltinISModuleGet(context, &mopts, &popts, &is, &curve_module));

        auto make = [&](OptixProgramGroupDesc desc) {
            OptixProgramGroupOptions gopts = {};
            OptixProgramGroup g = nullptr;
            size_t n = sizeof(log);
            FS_OPTIX(optixProgramGroupCreate(context, &desc, 1, &gopts, log, &n, &g));
            groups.push_back(g);
            return g;
        };
        OptixProgramGroupDesc d = {};
        d.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        d.raygen.module = module;
        d.raygen.entryFunctionName = "__raygen__camera";
        OptixProgramGroup raygen = make(d);

        d = {};
        d.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        d.miss.module = module;
        d.miss.entryFunctionName = "__miss__radiance";
        OptixProgramGroup miss_radiance = make(d);
        d.miss.entryFunctionName = "__miss__shadow";
        OptixProgramGroup miss_shadow = make(d);

        d = {};
        d.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        d.hitgroup.moduleCH = module;
        d.hitgroup.entryFunctionNameCH = "__closesthit__triangle";
        OptixProgramGroup tri_radiance = make(d);
        d = {};
        d.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        OptixProgramGroup tri_shadow = make(d);
        d = {};
        d.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        d.hitgroup.moduleCH = module;
        d.hitgroup.entryFunctionNameCH = "__closesthit__curve";
        d.hitgroup.moduleIS = curve_module;
        OptixProgramGroup curve_radiance = make(d);
        d = {};
        d.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        d.hitgroup.moduleIS = curve_module;
        OptixProgramGroup curve_shadow = make(d);

        OptixPipelineLinkOptions lopts = {};
        lopts.maxTraceDepth = 2;  // camera ray → shadow ray
        log_size = sizeof(log);
        FS_OPTIX(optixPipelineCreate(context, &popts, &lopts, groups.data(), static_cast<unsigned int>(groups.size()),
                                     log, &log_size, &pipeline));
        OptixStackSizes ss = {};
        for (auto g : groups)
            FS_OPTIX(optixUtilAccumulateStackSizes(g, &ss, pipeline));
        unsigned int dc_trav, dc_state, cont;
        FS_OPTIX(optixUtilComputeStackSizes(&ss, 2, 0, 0, &dc_trav, &dc_state, &cont));
        FS_OPTIX(optixPipelineSetStackSize(pipeline, dc_trav, dc_state, cont, 2));

        Record<Empty> rg;
        FS_OPTIX(optixSbtRecordPackHeader(raygen, &rg));
        raygen_rec.upload(&rg, sizeof(rg), stream);
        Record<Empty> ms[2];
        FS_OPTIX(optixSbtRecordPackHeader(miss_radiance, &ms[0]));
        FS_OPTIX(optixSbtRecordPackHeader(miss_shadow, &ms[1]));
        miss_rec.upload(ms, sizeof(ms), stream);
        // Instance 0 (triangles) uses records 0‥1, instance 1 (curves) 2‥3.
        Record<Empty> hg[4];
        FS_OPTIX(optixSbtRecordPackHeader(tri_radiance, &hg[0]));
        FS_OPTIX(optixSbtRecordPackHeader(tri_shadow, &hg[1]));
        FS_OPTIX(optixSbtRecordPackHeader(curve_radiance, &hg[2]));
        FS_OPTIX(optixSbtRecordPackHeader(curve_shadow, &hg[3]));
        hit_rec.upload(hg, sizeof(hg), stream);
        sbt.raygenRecord = raygen_rec.d();
        sbt.missRecordBase = miss_rec.d();
        sbt.missRecordStrideInBytes = sizeof(Record<Empty>);
        sbt.missRecordCount = 2;
        sbt.hitgroupRecordBase = hit_rec.d();
        sbt.hitgroupRecordStrideInBytes = sizeof(Record<Empty>);
        sbt.hitgroupRecordCount = 4;
        FS_CUDA(cudaStreamSynchronize(stream));
    }

    OptixTraversableHandle build_gas(const OptixBuildInput& input, Buffer& out_buf)
    {
        OptixAccelBuildOptions opts = {};
        opts.buildFlags = BUILD_FLAGS;
        opts.operation = OPTIX_BUILD_OPERATION_BUILD;
        OptixAccelBufferSizes sizes;
        FS_OPTIX(optixAccelComputeMemoryUsage(context, &opts, &input, 1, &sizes));
        temp.reserve(sizes.tempSizeInBytes);
        out_buf.reserve(sizes.outputSizeInBytes);
        OptixTraversableHandle h = 0;
        FS_OPTIX(optixAccelBuild(context, stream, &opts, &input, 1, temp.d(), temp.size, out_buf.d(),
                                 out_buf.size, &h, nullptr, 0));
        FS_CUDA(cudaStreamSynchronize(stream));
        return h;
    }

    void set_scene(const float* verts, unsigned int ntri, const unsigned int* tri_mat, const float* mats,
                   unsigned int nmat, const float* trace_verts, unsigned int nverts, const unsigned int* seg_index,
                   const float* seg_info, unsigned int nsegs)
    {
        static_assert(sizeof(Material) == 16 * sizeof(float), "Material layout");
        vertices.upload(verts, sizeof(float) * 9 * ntri, stream);
        tri_material.upload(tri_mat, sizeof(unsigned int) * ntri, stream);
        materials.upload(mats, sizeof(Material) * nmat, stream);
        num_materials = nmat;

        OptixBuildInput tri = {};
        tri.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        CUdeviceptr vbuf = vertices.d();
        tri.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        tri.triangleArray.vertexStrideInBytes = sizeof(float3);
        tri.triangleArray.numVertices = ntri * 3;
        tri.triangleArray.vertexBuffers = &vbuf;
        unsigned int tri_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
        tri.triangleArray.flags = &tri_flags;
        tri.triangleArray.numSbtRecords = 1;
        tri_handle = build_gas(tri, tri_gas);

        // board traces: straight round segments, (x, y, z, radius) per vertex
        std::vector<float3> pts(nverts);
        std::vector<float> widths(nverts);
        for (unsigned int i = 0; i < nverts; ++i)
        {
            pts[i] = make_float3(trace_verts[i * 4], trace_verts[i * 4 + 1], trace_verts[i * 4 + 2]);
            widths[i] = trace_verts[i * 4 + 3];
        }
        curve_handle = 0;
        if (nsegs)
        {
            curve_points.upload(pts.data(), pts.size() * sizeof(float3), stream);
            curve_widths.upload(widths.data(), widths.size() * sizeof(float), stream);
            curve_index.upload(seg_index, nsegs * sizeof(unsigned int), stream);
            segments.upload(seg_info, nsegs * sizeof(float4), stream);
            OptixBuildInput cin = {};
            cin.type = OPTIX_BUILD_INPUT_TYPE_CURVES;
            CUdeviceptr pbuf = curve_points.d(), wbuf = curve_widths.d();
            cin.curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR;
            cin.curveArray.numPrimitives = nsegs;
            cin.curveArray.vertexBuffers = &pbuf;
            cin.curveArray.numVertices = nverts;
            cin.curveArray.vertexStrideInBytes = sizeof(float3);
            cin.curveArray.widthBuffers = &wbuf;
            cin.curveArray.widthStrideInBytes = sizeof(float);
            cin.curveArray.indexBuffer = curve_index.d();
            cin.curveArray.indexStrideInBytes = sizeof(unsigned int);
            cin.curveArray.flag = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
            curve_handle = build_gas(cin, curve_gas);
        }

        std::vector<OptixInstance> inst;
        const float identity[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
        for (int k = 0; k < 2; ++k)
        {
            OptixTraversableHandle h = k == 0 ? tri_handle : curve_handle;
            if (!h)
                continue;
            OptixInstance oi = {};
            memcpy(oi.transform, identity, sizeof(identity));
            oi.instanceId = k;
            oi.sbtOffset = k * RAY_COUNT;
            oi.visibilityMask = 255;
            oi.flags = OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
            oi.traversableHandle = h;
            inst.push_back(oi);
        }
        instances.upload(inst.data(), inst.size() * sizeof(OptixInstance), stream);
        OptixBuildInput iin = {};
        iin.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        iin.instanceArray.instances = instances.d();
        iin.instanceArray.numInstances = static_cast<unsigned int>(inst.size());
        OptixAccelBuildOptions opts = {};
        opts.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        opts.operation = OPTIX_BUILD_OPERATION_BUILD;
        OptixAccelBufferSizes sizes;
        FS_OPTIX(optixAccelComputeMemoryUsage(context, &opts, &iin, 1, &sizes));
        temp.reserve(sizes.tempSizeInBytes);
        ias.reserve(sizes.outputSizeInBytes);
        FS_OPTIX(optixAccelBuild(context, stream, &opts, &iin, 1, temp.d(), temp.size, ias.d(), ias.size, &top,
                                 nullptr, 0));
        FS_CUDA(cudaStreamSynchronize(stream));
    }

    void resize(unsigned int w, unsigned int h)
    {
        if (w == width && h == height)
            return;
        width = w;
        height = h;
        size_t n = static_cast<size_t>(w) * h;
        color.reserve(n * sizeof(float4));
        albedo.reserve(n * sizeof(float4));
        normal.reserve(n * sizeof(float4));
        denoised.reserve(n * sizeof(float4));
        out.reserve(n * sizeof(uchar4));
        size_t nb = static_cast<size_t>((w + 1) / 2) * ((h + 1) / 2);
        bloom_a.reserve(nb * sizeof(float4));
        bloom_b.reserve(nb * sizeof(float4));
        OptixDenoiserSizes ds;
        FS_OPTIX(optixDenoiserComputeMemoryResources(denoiser, w, h, &ds));
        den_state.reserve(ds.stateSizeInBytes);
        den_scratch_size = ds.withoutOverlapScratchSizeInBytes;
        den_scratch.reserve(den_scratch_size);
        den_intensity.reserve(sizeof(float));
        den_avg.reserve(3 * sizeof(float));
        FS_OPTIX(optixDenoiserSetup(denoiser, stream, w, h, den_state.d(), ds.stateSizeInBytes, den_scratch.d(),
                                    den_scratch_size));
    }

    OptixImage2D image(const Buffer& b) const
    {
        OptixImage2D im = {};
        im.data = b.d();
        im.width = width;
        im.height = height;
        im.rowStrideInBytes = width * sizeof(float4);
        im.pixelStrideInBytes = sizeof(float4);
        im.format = OPTIX_PIXEL_FORMAT_FLOAT4;
        return im;
    }

    void render(const float* cam, unsigned int w, unsigned int h, unsigned int spp, unsigned int depth, int use_denoiser,
                float exposure, unsigned char* dst)
    {
        if (!top)
            throw std::runtime_error("no scene");
        resize(w, h);
        LaunchParams lp = {};
        lp.color = static_cast<float4*>(color.ptr);
        lp.albedo = static_cast<float4*>(albedo.ptr);
        lp.normal = static_cast<float4*>(normal.ptr);
        lp.width = w;
        lp.height = h;
        lp.spp = spp;
        lp.max_depth = depth;
        lp.eye = make_float3(cam[0], cam[1], cam[2]);
        lp.focal = cam[3];
        lp.fwd = make_float3(cam[4], cam[5], cam[6]);
        lp.cx = cam[7];
        lp.right = make_float3(cam[8], cam[9], cam[10]);
        lp.cy = cam[11];
        lp.up = make_float3(cam[12], cam[13], cam[14]);
        lp.time = cam[15];
        lp.handle = top;
        lp.materials = static_cast<const Material*>(materials.ptr);
        lp.tri_material = static_cast<const unsigned int*>(tri_material.ptr);
        lp.pipes = static_cast<const PipeParams*>(pipes.ptr);
        lp.segments = static_cast<const float4*>(segments.ptr);
        lp.num_pipes = num_pipes;
        lp.num_materials = num_materials;
        params_buf.upload(&lp, sizeof(lp), stream);
        FS_OPTIX(optixLaunch(pipeline, stream, params_buf.d(), sizeof(LaunchParams), &sbt, w, h, 1));

        const Buffer* result = &color;
        if (use_denoiser)
        {
            OptixImage2D in = image(color);
            FS_OPTIX(optixDenoiserComputeIntensity(denoiser, stream, &in, den_intensity.d(), den_scratch.d(),
                                                   den_scratch_size));
            FS_OPTIX(optixDenoiserComputeAverageColor(denoiser, stream, &in, den_avg.d(), den_scratch.d(),
                                                      den_scratch_size));
            OptixDenoiserParams dp = {};
            dp.hdrIntensity = den_intensity.d();
            dp.hdrAverageColor = den_avg.d();
            OptixDenoiserGuideLayer guide = {};
            guide.albedo = image(albedo);
            guide.normal = image(normal);
            OptixDenoiserLayer layer = {};
            layer.input = in;
            layer.output = image(denoised);
            FS_OPTIX(optixDenoiserInvoke(denoiser, stream, &dp, den_state.d(), den_state.size, &guide, &layer, 1, 0,
                                         0, den_scratch.d(), den_scratch_size));
            result = &denoised;
        }
        dim3 block(16, 16), grid((w + 15) / 16, (h + 15) / 16);
        const unsigned int bw = (w + 1) / 2, bh = (h + 1) / 2;
        dim3 bgrid((bw + 15) / 16, (bh + 15) / 16);
        auto* src = static_cast<const float4*>(result->ptr);
        auto* ba = static_cast<float4*>(bloom_a.ptr);
        auto* bb = static_cast<float4*>(bloom_b.ptr);
        bright_pass<<<bgrid, block, 0, stream>>>(src, ba, w, h, bw, bh);
        for (int pass = 0; pass < 2; ++pass)  // two separable passes ≈ a wide Gaussian
        {
            blur<<<bgrid, block, 0, stream>>>(ba, bb, bw, bh, 1, 0);
            blur<<<bgrid, block, 0, stream>>>(bb, ba, bw, bh, 0, 1);
        }
        tonemap<<<grid, block, 0, stream>>>(src, ba, static_cast<uchar4*>(out.ptr), w, h, bw, bh, exposure);
        FS_CUDA(cudaGetLastError());
        FS_CUDA(cudaMemcpyAsync(dst, out.ptr, static_cast<size_t>(w) * h * 4, cudaMemcpyDeviceToHost, stream));
        FS_CUDA(cudaStreamSynchronize(stream));
    }
};

// ------------------------------------------------------------------ C API
extern "C" {

__attribute__((visibility("default"))) void* fs_create(const char* module_path, char* err, int errlen)
{
    try
    {
        return new FsRenderer(module_path);
    }
    catch (const std::exception& e)
    {
        set_error(e, err, errlen);
        return nullptr;
    }
}

__attribute__((visibility("default"))) void fs_destroy(void* r)
{
    delete static_cast<FsRenderer*>(r);
}

__attribute__((visibility("default"))) const char* fs_device_name(void* r)
{
    return static_cast<FsRenderer*>(r)->device_name.c_str();
}

__attribute__((visibility("default"))) int fs_set_scene(void* r, const float* verts, unsigned int ntri,
                                                        const unsigned int* tri_mat, const float* mats,
                                                        unsigned int nmat, const float* trace_verts,
                                                        unsigned int nverts, const unsigned int* seg_index,
                                                        const float* seg_info, unsigned int nsegs, char* err,
                                                        int errlen)
{
    try
    {
        static_cast<FsRenderer*>(r)->set_scene(verts, ntri, tri_mat, mats, nmat, trace_verts, nverts, seg_index,
                                               seg_info, nsegs);
        return 0;
    }
    catch (const std::exception& e)
    {
        set_error(e, err, errlen);
        return -1;
    }
}

__attribute__((visibility("default"))) int fs_set_pipes(void* r, const float* data, unsigned int n, char* err,
                                                        int errlen)
{
    try
    {
        static_assert(sizeof(PipeParams) == 12 * sizeof(float), "PipeParams layout");
        auto* R = static_cast<FsRenderer*>(r);
        R->pipes.upload(data, sizeof(PipeParams) * n, R->stream);
        R->num_pipes = n;
        return 0;
    }
    catch (const std::exception& e)
    {
        set_error(e, err, errlen);
        return -1;
    }
}

__attribute__((visibility("default"))) int fs_render(void* r, const float* camera, unsigned int w, unsigned int h,
                                                     unsigned int spp, unsigned int depth, int denoise, float exposure,
                                                     unsigned char* out_bgrx, char* err, int errlen)
{
    try
    {
        static_cast<FsRenderer*>(r)->render(camera, w, h, spp, depth, denoise, exposure, out_bgrx);
        return 0;
    }
    catch (const std::exception& e)
    {
        set_error(e, err, errlen);
        return -1;
    }
}

}  // extern "C"

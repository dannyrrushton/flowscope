// OptiX device programs: a small path tracer (next-event estimation for the key
// light, one or two diffuse/specular bounces that pick up the glowing pipes) with
// procedural materials for the motherboard parts.
#include <optix.h>

#include "shared.h"
#include "vec.h"

using namespace fs;

extern "C" __constant__ LaunchParams params;

// ------------------------------------------------------------------ lighting
// Soft key light: a disc above and behind the rear I/O; plus a cool fill disc.
__constant__ float3 KEY_POS = {-14.0f, 46.0f, -22.0f};
__constant__ float KEY_RADIUS = 9.0f;
__constant__ float3 KEY_COLOR = {1.0f, 0.95f, 0.88f};
__constant__ float KEY_POWER = 150.0f;
__constant__ float3 FILL_POS = {30.0f, 26.0f, 30.0f};
__constant__ float FILL_RADIUS = 12.0f;
__constant__ float3 FILL_COLOR = {0.55f, 0.65f, 0.85f};
__constant__ float FILL_POWER = 75.0f;

static __forceinline__ __device__ float3 environment(float3 d)
{
    // dark studio: slightly lighter toward the zenith, warm bounce near the floor
    float t = 0.5f * (d.y + 1.0f);
    float3 top = make_float3(0.36f, 0.39f, 0.44f);
    float3 low = make_float3(0.10f, 0.095f, 0.09f);
    return lerp(low, top, t * t);
}

// Volume ramp (0‥1): navy → blue → azure → ice → white, rising in lightness.
// Matches HEAT_STOPS in scene3d.py. Returned in linear RGB.
static __forceinline__ __device__ float3 heat_color(float t)
{
    const float3 stops[5] = {
        make_float3(0.04f, 0.12f, 0.36f), make_float3(0.12f, 0.37f, 0.88f),
        make_float3(0.18f, 0.63f, 0.96f), make_float3(0.56f, 0.86f, 1.00f),
        make_float3(0.94f, 0.98f, 1.00f)};
    const float pos[5] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    t = clampf(t, 0.0f, 1.0f);
    float3 c = stops[4];
    for (int i = 0; i < 4; ++i)
    {
        if (t <= pos[i + 1])
        {
            float f = (t - pos[i]) / (pos[i + 1] - pos[i]);
            c = lerp(stops[i], stops[i + 1], f);
            break;
        }
    }
    return srgb_to_linear(c);
}

// Power ramp (0‥1): dark red → red → salmon → pale pink, rising in lightness.
// Matches POWER_STOPS["dark"] in scene3d.py. Returned in linear RGB.
static __forceinline__ __device__ float3 power_color(float t)
{
    const float3 stops[5] = {
        make_float3(0.24f, 0.03f, 0.03f), make_float3(0.56f, 0.07f, 0.07f),
        make_float3(0.83f, 0.14f, 0.14f), make_float3(1.00f, 0.43f, 0.43f),
        make_float3(1.00f, 0.84f, 0.84f)};
    t = clampf(t, 0.0f, 1.0f) * 4.0f;
    int i = min(static_cast<int>(t), 3);
    return srgb_to_linear(lerp(stops[i], stops[i + 1], t - i));
}

static __forceinline__ __device__ float glow_intensity(float heat)
{
    return heat <= 0.0f ? 0.0f : 0.6f + 9.5f * powf(heat, 1.5f);
}

// ------------------------------------------------------------------ payload
struct PRD
{
    float3 radiance;
    float3 throughput;
    float3 origin;
    float3 direction;
    float3 albedo;  // first hit, for the denoiser
    float3 normal;
    unsigned int seed;
    int depth;
    int done;
};

static __forceinline__ __device__ void* unpack_ptr(unsigned int i0, unsigned int i1)
{
    const unsigned long long p = static_cast<unsigned long long>(i0) << 32 | i1;
    return reinterpret_cast<void*>(p);
}

static __forceinline__ __device__ void pack_ptr(void* ptr, unsigned int& i0, unsigned int& i1)
{
    const unsigned long long p = reinterpret_cast<unsigned long long>(ptr);
    i0 = p >> 32;
    i1 = p & 0x00000000ffffffffull;
}

static __forceinline__ __device__ PRD* prd()
{
    return reinterpret_cast<PRD*>(unpack_ptr(optixGetPayload_0(), optixGetPayload_1()));
}

static __forceinline__ __device__ bool occluded(float3 from, float3 to)
{
    float3 d = to - from;
    float dist = length(d);
    unsigned int hit = 1;
    optixTrace(params.handle, from, d / dist, 0.0f, dist - 1e-3f, 0.0f, OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT |
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT,
               RAY_SHADOW, RAY_COUNT, RAY_SHADOW, hit);
    return hit != 0;
}

// Direct light from one disc light (diffuse + a Blinn-Phong lobe).
static __forceinline__ __device__ float3 disc_light(float3 P, float3 N, float3 V, float3 albedo,
                                                    float rough, float metal, float3 pos, float radius,
                                                    float3 color, float power, unsigned int& seed)
{
    float r = radius * sqrtf(rnd(seed));
    float a = 2.0f * M_PIf * rnd(seed);
    float3 Lp = pos + make_float3(r * cosf(a), 0.0f, r * sinf(a));
    float3 L = Lp - P;
    float d2 = dot(L, L);
    L = L / sqrtf(d2);
    float ndl = dot(N, L);
    if (ndl <= 0.0f)
        return make_float3(0.0f);
    float cos_l = fmaxf(-L.y, 0.0f);  // disc faces down
    if (cos_l <= 0.0f || occluded(P + N * 2e-3f, Lp))
        return make_float3(0.0f);
    float area = M_PIf * radius * radius;
    float3 E = color * (power * ndl * cos_l * area / d2 / 100.0f);
    float3 diffuse = albedo * ((1.0f - metal) / M_PIf);
    float3 H = normalize(L + V);
    float shin = fmaxf(2.0f / fmaxf(rough * rough, 1e-3f) - 2.0f, 1.0f);
    float3 F0 = lerp(make_float3(0.04f), albedo, metal);
    float spec_n = (shin + 2.0f) / (8.0f * M_PIf) * powf(fmaxf(dot(N, H), 0.0f), shin);
    return E * (diffuse + F0 * spec_n);
}

// ------------------------------------------------------------------ surfaces
struct Surface
{
    float3 albedo;
    float3 emission;
    float roughness;
    float metallic;
};

static __forceinline__ __device__ float pipe_heat(float slot)
{
    int i = static_cast<int>(slot);
    if (i < 0 || i >= static_cast<int>(params.num_pipes))
        return 0.0f;
    return params.pipes[i].heat;
}

// 1 for a measured power slot, dimmer for an estimate.
static __forceinline__ __device__ float power_weight(int slot)
{
    if (slot < 0 || slot >= static_cast<int>(params.num_pipes))
        return 0.0f;
    return (static_cast<int>(params.pipes[slot].flags) & PIPE_ESTIMATE) ? 0.45f : 1.0f;
}

// Red power glow for a surface: a hotspot over `rect` on top faces, a dimmer
// even glow on the sides. Returns 0 when the power layer is off.
static __forceinline__ __device__ float3 power_glow(int slot, const Material& m, bool top, float lx, float lz)
{
    float ph = pipe_heat(static_cast<float>(slot));
    if (ph <= 0.0f)
        return make_float3(0.0f);
    float g = 0.35f;
    if (top && m.rect.z > 0.0f && m.rect.w > 0.0f)
        g = expf(-1.6f * ((lx * lx) / (m.rect.z * m.rect.z) + (lz * lz) / (m.rect.w * m.rect.w)));
    return power_color(ph * (0.5f + 0.5f * g)) * (glow_intensity(ph) * 0.5f * g * power_weight(slot));
}

static __device__ Surface evaluate(const Material& m, float3 P, float3 N)
{
    Surface s;
    s.albedo = m.albedo;
    s.emission = make_float3(0.0f);
    s.roughness = m.roughness;
    s.metallic = m.metallic;
    const int pattern = static_cast<int>(m.pattern);
    const bool top = N.y > 0.9f;
    const float lx = P.x - m.rect.x, lz = P.z - m.rect.y;  // pattern-local

    switch (pattern)
    {
    case PAT_PCB:
    {
        if (!top)
        {
            s.albedo = make_float3(0.32f, 0.27f, 0.17f);  // FR-4 edge
            s.roughness = 0.8f;
            break;
        }
        // 0.25 cm routing grid; each cell carries a horizontal or vertical trace
        const float cell = 0.25f;
        float gx = floorf(P.x / cell), gz = floorf(P.z / cell);
        float fx = P.x / cell - gx, fz = P.z / cell - gz;
        float h = hash2(gx * 0.37f + 3.1f, gz * 0.61f + 7.7f);
        float row = hash2(0.0f, gz);  // long horizontal buses
        float col = hash2(gx, 0.0f);
        float trace = 0.0f;
        if (row > 0.55f && fabsf(fz - 0.5f) < 0.12f)
            trace = 1.0f;
        if (col > 0.72f && fabsf(fx - 0.5f) < 0.10f && h > 0.3f)
            trace = 1.0f;
        float via = 0.0f;
        if (h > 0.93f)
        {
            float d = sqrtf((fx - 0.5f) * (fx - 0.5f) + (fz - 0.5f) * (fz - 0.5f));
            via = d < 0.16f ? 1.0f : 0.0f;
            if (d < 0.07f)
                via = 2.0f;  // drilled hole
        }
        float3 mask = m.albedo;
        s.albedo = lerp(mask, mask * 1.55f + make_float3(0.01f, 0.02f, 0.0f), trace);
        s.roughness = trace > 0.0f ? 0.28f : m.roughness;
        if (via >= 1.0f)
        {
            s.albedo = via > 1.5f ? make_float3(0.02f) : make_float3(0.75f, 0.62f, 0.35f);
            s.metallic = via > 1.5f ? 0.0f : 1.0f;
            s.roughness = 0.3f;
        }
        // silkscreen: short white marks along component outlines (extra.x = density)
        float sh = hash2(floorf(P.x / 0.9f) + 11.0f, floorf(P.z / 0.18f) - 5.0f);
        if (sh > 0.985f && fmodf(fabsf(P.x), 0.9f) < 0.55f && fmodf(fabsf(P.z), 0.18f) < 0.05f)
        {
            s.albedo = make_float3(0.85f);
            s.metallic = 0.0f;
            s.roughness = 0.7f;
        }
        break;
    }
    case PAT_BRUSHED:
    case PAT_FINS:
    {
        float streak = hash2(floorf(P.z * 180.0f), 0.0f) * 0.12f + hash2(floorf(P.x * 3.0f), floorf(P.z * 40.0f)) * 0.05f;
        s.albedo = m.albedo * (0.9f + streak);
        break;
    }
    case PAT_CHIP:
    {
        float heat = pipe_heat(m.heat_slot);
        if (top)
        {
            // laser-etched pin-1 dot and faint marking lines
            float hx = m.rect.z, hz = m.rect.w;
            float px = lx + hx * 0.7f, pz = lz + hz * 0.7f;
            if (px * px + pz * pz < 0.012f)
                s.albedo = m.albedo * 0.45f;
            else if (fabsf(lz) < hz * 0.35f && fmodf(fabsf(lx) + 0.03f, 0.12f) < 0.08f &&
                     hash2(floorf(lx / 0.12f), floorf(lz / 0.09f)) > 0.35f &&
                     fmodf(fabsf(lz), 0.09f) < 0.035f)
                s.albedo = m.albedo * 2.2f + make_float3(0.03f);
            // thermal gradient: a hotspot in the middle of the die
            if (heat > 0.0f)
            {
                float r2 = (lx * lx) / (hx * hx) + (lz * lz) / (hz * hz);
                float g = expf(-2.2f * r2);
                s.emission = heat_color(heat * (0.45f + 0.55f * g)) * (glow_intensity(heat) * 0.5f * g);
            }
            // with the power layer on, a powered chip shows its watts instead
            int ps = static_cast<int>(m.extra.w) - 1;
            if (ps >= 0 && pipe_heat(static_cast<float>(ps)) > 0.0f)
                s.emission = power_glow(ps, m, true, lx, lz);
        }
        break;
    }
    case PAT_POWER:
        s.emission = power_glow(static_cast<int>(m.heat_slot), m, top, lx, lz);
        break;
    case PAT_GOLD:
    {
        // gold contact fingers along x (extra.x = pitch)
        float pitch = m.extra.x > 0.0f ? m.extra.x : 0.1f;
        if (fmodf(fabsf(lx), pitch) < pitch * 0.65f)
        {
            s.albedo = make_float3(1.0f, 0.78f, 0.34f);
            s.metallic = 1.0f;
            s.roughness = 0.25f;
        }
        break;
    }
    case PAT_LABEL:
    {
        if (!top)
            break;
        float hx = m.rect.z, hz = m.rect.w;
        float u = (lx / hx) * 0.5f + 0.5f, v = (lz / hz) * 0.5f + 0.5f;
        if (u > 0.06f && u < 0.94f && v > 0.08f && v < 0.92f)
        {
            s.albedo = make_float3(m.extra.x, m.extra.y, m.extra.z);  // sticker
            s.metallic = 0.0f;
            s.roughness = 0.55f;
            float line = floorf(v * 7.0f);
            float fv = v * 7.0f - line;
            float word = hash2(floorf(u * 18.0f), line);
            if (line >= 1.0f && line <= 5.0f && fv > 0.3f && fv < 0.7f && word > 0.3f &&
                u < 0.25f + 0.6f * hash2(line, 3.0f))
                s.albedo = make_float3(0.05f);
            if (u < 0.2f && v > 0.2f && v < 0.8f)  // colored brand band
                s.albedo = make_float3(m.extra.w, m.extra.w * 0.45f, 0.08f);  // warm: blue means data volume
        }
        break;
    }
    case PAT_FLOOR:
    {
        float n = hash2(floorf(P.x * 60.0f), floorf(P.z * 60.0f));
        s.albedo = m.albedo * (0.92f + 0.16f * n);
        break;
    }
    case PAT_GLOW:
    {
        float heat = pipe_heat(m.heat_slot);
        float k = heat > 0.0f ? 1.5f + 10.0f * heat : 0.02f;
        s.emission = make_float3(m.extra.x, m.extra.y, m.extra.z) * k;
        break;
    }
    default:
        break;
    }
    if (m.emission > 0.0f)
        s.emission += s.albedo * m.emission;
    return s;
}

// Scatter the path from a surface: accumulate emission + direct light, pick
// the next bounce direction.
static __device__ void scatter(PRD& p, float3 P, float3 N, const Surface& s)
{
    const float3 V = -p.direction;
    if (p.depth == 0)
    {
        p.albedo = s.albedo + s.emission;
        p.normal = N;
    }
    float3 emitted = s.emission;
    if (p.depth > 0)  // tame fireflies from small hot emitters
        emitted = fminf(emitted, make_float3(14.0f));
    p.radiance += p.throughput * emitted;

    float3 direct = disc_light(P, N, V, s.albedo, s.roughness, s.metallic, KEY_POS, KEY_RADIUS, KEY_COLOR,
                               KEY_POWER, p.seed) +
                    disc_light(P, N, V, s.albedo, s.roughness, s.metallic, FILL_POS, FILL_RADIUS, FILL_COLOR,
                               FILL_POWER, p.seed);
    p.radiance += p.throughput * direct;

    float spec_p = 0.12f + 0.78f * s.metallic;
    float3 F0 = lerp(make_float3(0.04f), s.albedo, s.metallic);
    float3 dir;
    if (rnd(p.seed) < spec_p)
    {
        float3 R = reflect(-V, N);
        float r = s.roughness * s.roughness;
        dir = normalize(R + random_in_sphere(p.seed) * r);
        if (dot(dir, N) <= 0.0f)
            dir = R;
        float fres = 1.0f - fmaxf(dot(N, V), 0.0f);
        float3 F = F0 + (make_float3(1.0f) - F0) * (fres * fres * fres * fres * fres);
        p.throughput *= F / spec_p;
    }
    else
    {
        dir = cosine_hemisphere(N, p.seed);
        p.throughput *= s.albedo * (1.0f - s.metallic) / (1.0f - spec_p);
    }
    p.origin = P + N * 2e-3f;
    p.direction = dir;
}

// ------------------------------------------------------------------ programs
extern "C" __global__ void __raygen__camera()
{
    const uint3 idx = optixGetLaunchIndex();
    const unsigned int pixel = idx.y * params.width + idx.x;
    // Fixed per-pixel seed: the noise pattern stays still between frames, so the
    // denoised image doesn't shimmer while pipes animate.
    unsigned int seed = tea(pixel, 0x9e37u);
    float3 sum = make_float3(0.0f), albedo = make_float3(0.0f), normal = make_float3(0.0f);

    for (unsigned int s = 0; s < params.spp; ++s)
    {
        float px = idx.x + rnd(seed), py = idx.y + rnd(seed);
        PRD p;
        p.radiance = make_float3(0.0f);
        p.throughput = make_float3(1.0f);
        p.origin = params.eye;
        p.direction = normalize(params.fwd + params.right * ((px - params.cx) / params.focal) -
                                params.up * ((py - params.cy) / params.focal));
        p.albedo = make_float3(0.0f);
        p.normal = make_float3(0.0f);
        p.seed = seed;
        p.done = 0;
        unsigned int u0, u1;
        pack_ptr(&p, u0, u1);
        for (p.depth = 0; p.depth < static_cast<int>(params.max_depth) && !p.done; ++p.depth)
        {
            optixTrace(params.handle, p.origin, p.direction, 1e-4f, 1e16f, 0.0f, OptixVisibilityMask(255),
                       OPTIX_RAY_FLAG_DISABLE_ANYHIT, RAY_RADIANCE, RAY_COUNT, RAY_RADIANCE, u0, u1);
            float mx = fmaxf(p.throughput.x, fmaxf(p.throughput.y, p.throughput.z));
            if (mx < 0.01f)
                break;
        }
        seed = p.seed;
        sum += p.radiance;
        albedo += p.albedo;
        normal += p.normal;
    }
    const float inv = 1.0f / params.spp;
    params.color[pixel] = make_float4(sum * inv, 1.0f);
    params.albedo[pixel] = make_float4(albedo * inv, 1.0f);
    params.normal[pixel] = make_float4(normal * inv, 0.0f);
}

extern "C" __global__ void __miss__radiance()
{
    PRD& p = *prd();
    float3 env = environment(p.direction);
    if (p.depth == 0)
        p.albedo = env;
    p.radiance += p.throughput * env;
    p.done = 1;
}

extern "C" __global__ void __miss__shadow()
{
    optixSetPayload_0(0);
}

extern "C" __global__ void __closesthit__triangle()
{
    PRD& p = *prd();
    float3 v[3];
    optixGetTriangleVertexData(v);
    float3 N = normalize(cross(v[1] - v[0], v[2] - v[0]));
    if (dot(N, p.direction) > 0.0f)
        N = -N;
    const float3 P = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
    const unsigned int prim = optixGetPrimitiveIndex();
    const Material m = params.materials[params.tri_material[prim]];
    scatter(p, P, N, evaluate(m, P, N));
}

// Pulses travelling along a trace: returns 0‥1 coverage at parameter x.
static __forceinline__ __device__ float streak(float x, float count, float length)
{
    if (count <= 0.0f)
        return 0.0f;
    float f = x * count;
    f -= floorf(f);
    float d = fminf(f, 1.0f - f) * length / count;  // cm to the nearest pulse
    return clampf(1.0f - d / 0.3f, 0.0f, 1.0f);
}

// Board traces: tin-plated copper when idle; when data flows they glow along
// the heat ramp by their own direction's volume, with a slow wave and bright
// pulses running toward the CPU (in) or away from it (out).
extern "C" __global__ void __closesthit__curve()
{
    PRD& p = *prd();
    float4 cp[2];
    optixGetLinearCurveVertexData(cp);
    const float u = optixGetCurveParameter();
    const float3 A = make_float3(cp[0]), B = make_float3(cp[1]);
    const float3 C = A + (B - A) * u;
    const float3 P = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
    float3 N = P - C;
    N = dot(N, N) > 1e-10f ? normalize(N) : -p.direction;
    if (dot(N, p.direction) > 0.0f)
        N = -N;

    const float4 seg = params.segments[optixGetPrimitiveIndex()];
    const PipeParams pp = params.pipes[static_cast<int>(seg.x)];
    const int kind = static_cast<int>(seg.y);
    const float sp = seg.z + (seg.w - seg.z) * u;  // 0 = device end, 1 = CPU end
    const int flags = static_cast<int>(pp.flags);
    const float heat = kind == TRACE_IN ? pp.heat_in : kind == TRACE_OUT ? pp.heat_out : pp.heat;

    Surface s;
    s.emission = make_float3(0.0f);
    if (heat > 0.0f)
    {
        const float s_cm = sp * pp.length;
        float wave = 0.78f + 0.22f * sinf(6.2832f * (s_cm / 5.0f) - params.time * (1.0f + 5.0f * heat));
        float3 hot = heat_color(heat * (0.8f + 0.2f * wave));
        s.albedo = hot * 0.3f;
        s.metallic = 0.0f;
        s.roughness = 0.4f;
        s.emission = hot * (glow_intensity(heat) * wave);
        float pulse = kind == TRACE_OUT ? streak(1.0f - sp - pp.phase_out, pp.dots_out, pp.length)
                                        : streak(sp - pp.phase_in, pp.dots_in, pp.length);
        s.emission += make_float3(0.92f, 0.97f, 1.0f) * (4.0f * pulse * pulse);
    }
    else
    {
        s.albedo = make_float3(0.55f, 0.56f, 0.58f);
        s.metallic = 1.0f;
        s.roughness = 0.35f;
    }
    if (flags & PIPE_FADED)
    {
        s.emission *= 0.08f;
        s.albedo *= 0.5f;
    }
    if (flags & (PIPE_HOVER | PIPE_SELECTED))
        s.emission += make_float3(0.9f, 0.95f, 1.0f) * ((flags & PIPE_SELECTED) ? 2.5f : 1.2f);
    scatter(p, P, N, s);
}

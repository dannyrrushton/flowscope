// Types shared by the host library, the OptiX device programs and (by layout) the
// Python ctypes wrapper in flowscope/optix_backend.py. Keep the three in sync.
#pragma once

#include <optix.h>
#include <vector_types.h>

namespace fs
{

// Procedural surface patterns evaluated in the closest-hit program.
enum Pattern : int
{
    PAT_PLAIN = 0,
    PAT_PCB = 1,       // solder mask with copper traces, vias and silkscreen
    PAT_BRUSHED = 2,   // brushed / anodised metal
    PAT_CHIP = 3,      // epoxy package with a heat-map gradient on top
    PAT_FINS = 4,      // heatsink fins (stripes along x)
    PAT_GOLD = 5,      // gold edge-connector fingers
    PAT_LABEL = 6,     // printed sticker (SSD / DIMM labels)
    PAT_FLOOR = 7,     // case floor: dark powder-coated steel
    PAT_GLOW = 8,      // component glow driven by a heat slot (LED-ish)
    PAT_POWER = 9,     // part lit red by a power slot (heat_slot), gradient over rect
};

// 16 floats, matches `Material` in optix_backend.py.
struct Material
{
    float3 albedo;
    float roughness;
    float metallic;
    float emission;
    float pattern;    // Pattern (float so the struct is all floats)
    float heat_slot;  // pipe index whose heat tints this surface, -1 = none
    float4 rect;      // pattern frame: centre x, centre z, half-size x, half-size z
    float4 extra;     // pattern-specific; PAT_CHIP: x = no markings, w = power slot + 1
};

// Per pipe, updated every frame. 12 floats, built in scene3d._render_optix.
// Power readings use the same layout after the pipes: heat = 0‥1 on the
// power scale, flags may carry PIPE_ESTIMATE.
struct PipeParams
{
    float heat, heat_in, heat_out, radius;
    float phase_in, phase_out, dots_in, dots_out;
    float flags, length, pad0, pad1;
};

enum PipeFlags : int
{
    PIPE_HOVER = 1,
    PIPE_SELECTED = 2,
    PIPE_FADED = 4,
    PIPE_ESTIMATE = 8,  // power slot holds an estimate, not a measurement
};

// Which trace of a pipe a curve segment belongs to (TRACE_* in scene3d.py).
enum TraceKind : int
{
    TRACE_IN = 0,    // inbound half of a differential pair
    TRACE_OUT = 1,   // outbound half
    TRACE_BOTH = 2,  // single trace carrying everything
};

enum RayType : int
{
    RAY_RADIANCE = 0,
    RAY_SHADOW = 1,
    RAY_COUNT = 2,
};

struct LaunchParams
{
    float4* color;   // linear HDR radiance, averaged over spp
    float4* albedo;  // denoiser guide
    float4* normal;  // denoiser guide (world space)
    unsigned int width, height, spp, max_depth;

    float3 eye;
    float focal;
    float3 fwd;
    float cx;
    float3 right;
    float cy;
    float3 up;
    float time;

    OptixTraversableHandle handle;
    const Material* materials;
    const unsigned int* tri_material;  // one per triangle
    const PipeParams* pipes;
    const float4* segments;  // per trace segment: pipe, TraceKind, s0, s1
    unsigned int num_pipes;
    unsigned int num_materials;
};

}  // namespace fs

// Minimal float3 math and random numbers for the device programs.
#pragma once

#include <vector_functions.h>
#include <vector_types.h>

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

#define FS_DEV static __forceinline__ __device__

FS_DEV float3 make_float3(float s) { return ::make_float3(s, s, s); }
FS_DEV float3 make_float3(float4 v) { return ::make_float3(v.x, v.y, v.z); }
FS_DEV float4 make_float4(float3 v, float w) { return ::make_float4(v.x, v.y, v.z, w); }

FS_DEV float3 operator+(float3 a, float3 b) { return ::make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
FS_DEV float3 operator-(float3 a, float3 b) { return ::make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
FS_DEV float3 operator-(float3 a) { return ::make_float3(-a.x, -a.y, -a.z); }
FS_DEV float3 operator*(float3 a, float3 b) { return ::make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
FS_DEV float3 operator*(float3 a, float s) { return ::make_float3(a.x * s, a.y * s, a.z * s); }
FS_DEV float3 operator*(float s, float3 a) { return a * s; }
FS_DEV float3 operator/(float3 a, float s) { return a * (1.0f / s); }
FS_DEV float3 operator/(float3 a, float3 b) { return ::make_float3(a.x / b.x, a.y / b.y, a.z / b.z); }
FS_DEV void operator+=(float3& a, float3 b) { a = a + b; }
FS_DEV void operator*=(float3& a, float3 b) { a = a * b; }
FS_DEV void operator*=(float3& a, float s) { a = a * s; }

FS_DEV float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
FS_DEV float3 cross(float3 a, float3 b)
{
    return ::make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
FS_DEV float length(float3 a) { return sqrtf(dot(a, a)); }
FS_DEV float3 normalize(float3 a) { return a * rsqrtf(fmaxf(dot(a, a), 1e-20f)); }
FS_DEV float3 reflect(float3 i, float3 n) { return i - 2.0f * dot(n, i) * n; }
FS_DEV float3 lerp(float3 a, float3 b, float t) { return a + (b - a) * t; }
FS_DEV float clampf(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
FS_DEV float3 fminf(float3 a, float3 b) { return ::make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)); }

FS_DEV float3 srgb_to_linear(float3 c)
{
    return ::make_float3(powf(c.x, 2.2f), powf(c.y, 2.2f), powf(c.z, 2.2f));
}

// ------------------------------------------------------------------ random
FS_DEV unsigned int tea(unsigned int v0, unsigned int v1)
{
    unsigned int s0 = 0;
    for (int n = 0; n < 16; ++n)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }
    return v0;
}

FS_DEV float rnd(unsigned int& state)
{
    state = state * 1664525u + 1013904223u;
    return static_cast<float>(state & 0x00ffffffu) / static_cast<float>(0x01000000u);
}

// Deterministic 0‥1 hash of a 2D lattice point (for procedural textures).
FS_DEV float hash2(float x, float y)
{
    unsigned int h = static_cast<unsigned int>(static_cast<int>(x)) * 73856093u ^
                     static_cast<unsigned int>(static_cast<int>(y)) * 19349663u;
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    return static_cast<float>(h & 0x00ffffffu) / static_cast<float>(0x01000000u);
}

FS_DEV float3 random_in_sphere(unsigned int& seed)
{
    float z = 1.0f - 2.0f * rnd(seed);
    float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    float a = 2.0f * M_PIf * rnd(seed);
    float k = cbrtf(rnd(seed));
    return ::make_float3(r * cosf(a) * k, r * sinf(a) * k, z * k);
}

FS_DEV float3 cosine_hemisphere(float3 n, unsigned int& seed)
{
    float r = sqrtf(rnd(seed));
    float a = 2.0f * M_PIf * rnd(seed);
    float x = r * cosf(a), y = r * sinf(a), z = sqrtf(fmaxf(0.0f, 1.0f - x * x - y * y));
    float3 t = fabsf(n.x) > 0.5f ? ::make_float3(0.0f, 1.0f, 0.0f) : ::make_float3(1.0f, 0.0f, 0.0f);
    float3 b = normalize(cross(n, t));
    t = cross(b, n);
    return normalize(t * x + b * y + n * z);
}

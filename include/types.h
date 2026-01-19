#pragma once

#include <cuda_runtime.h>
#include <cmath>

// CUDA already provides make_float3 and make_float4
// We just need to add operators and utility functions

// Math operators for float3
__host__ __device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 operator-(const float3& a) {
    return make_float3(-a.x, -a.y, -a.z);
}

__host__ __device__ inline float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline float3 operator*(float s, const float3& a) {
    return a * s;
}

__host__ __device__ inline float3 operator/(const float3& a, float s) {
    return make_float3(a.x / s, a.y / s, a.z / s);
}

// Math operators for float4
__host__ __device__ inline float4 operator+(const float4& a, const float4& b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__host__ __device__ inline float4 operator*(const float4& a, float s) {
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

__host__ __device__ inline float4 operator*(float s, const float4& a) {
    return a * s;
}

// Math functions
__host__ __device__ inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float length(const float3& v) {
    return sqrtf(dot(v, v));
}

__host__ __device__ inline float3 normalize(const float3& v) {
    float len = length(v);
    return len > 0 ? v / len : make_float3(0, 0, 0);
}

__host__ __device__ inline float3 cross(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}

__host__ __device__ inline float clamp(float x, float min_val, float max_val) {
    return fminf(fmaxf(x, min_val), max_val);
}

__host__ __device__ inline float3 clamp(const float3& v, float min_val, float max_val) {
    return make_float3(clamp(v.x, min_val, max_val),
                       clamp(v.y, min_val, max_val),
                       clamp(v.z, min_val, max_val));
}

__host__ __device__ inline float3 abs(const float3& v) {
    return make_float3(fabsf(v.x), fabsf(v.y), fabsf(v.z));
}

__host__ __device__ inline float3 max(const float3& a, const float3& b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__host__ __device__ inline float3 reflect(const float3& I, const float3& N) {
    return I - 2.0f * dot(N, I) * N;
}

// Matrix structure
struct mat4 {
    float m[16];
    
    __host__ __device__ mat4() {
        for (int i = 0; i < 16; i++) m[i] = 0;
        m[0] = m[5] = m[10] = m[15] = 1; // Identity
    }
};

// Matrix operations
__host__ __device__ inline float4 operator*(const mat4& m, const float4& v) {
    return make_float4(
        m.m[0] * v.x + m.m[4] * v.y + m.m[8]  * v.z + m.m[12] * v.w,
        m.m[1] * v.x + m.m[5] * v.y + m.m[9]  * v.z + m.m[13] * v.w,
        m.m[2] * v.x + m.m[6] * v.y + m.m[10] * v.z + m.m[14] * v.w,
        m.m[3] * v.x + m.m[7] * v.y + m.m[11] * v.z + m.m[15] * v.w
    );
}

// Camera parameters
struct CameraParams {
    mat4 transform;
    float fov;
    int width;
    int height;
};

// Fractal parameters
struct FractalParams {
    int type;  // 0=Mandelbox, 1=Menger, 2=Sierpinski, etc.
    int iterations;
    float scale;
    float minRadius;
    float maxRadius;
    float3 foldRadius;
    float3 offset;
    float rotationAngle;
    float3 colorOrbit;
    
    // Interpolation support
    float morphFactor;  // 0.0 to 1.0 for morphing between fractals
    int targetType;     // Target fractal type for morphing
    float targetScale;
    int targetIterations;
};

// Rendering parameters
struct RenderParams {
    int maxSteps;
    float maxDist;
    float epsilon;
    float3 lightDir;
    float3 backgroundColor;
    float ambientStrength;
    float diffuseStrength;
    float specularStrength;
    int specularPower;
};

#include "../include/types.h"

// ==================== Folding Operations ====================

__device__ inline void boxFold(float4& z, const float3& r) {
    z.x = clamp(z.x, -r.x, r.x) * 2.0f - z.x;
    z.y = clamp(z.y, -r.y, r.y) * 2.0f - z.y;
    z.z = clamp(z.z, -r.z, r.z) * 2.0f - z.z;
}

__device__ inline void sphereFold(float4& z, float minR, float maxR) {
    float r2 = z.x * z.x + z.y * z.y + z.z * z.z;
    float minR2 = minR * minR;
    float maxR2 = maxR * maxR;
    
    if (r2 < minR2) {
        float factor = maxR2 / minR2;
        z.x *= factor;
        z.y *= factor;
        z.z *= factor;
        z.w *= factor;
    } else if (r2 < maxR2) {
        float factor = maxR2 / r2;
        z.x *= factor;
        z.y *= factor;
        z.z *= factor;
        z.w *= factor;
    }
}

__device__ inline void absFold(float4& z) {
    z.x = fabsf(z.x);
    z.y = fabsf(z.y);
    z.z = fabsf(z.z);
}

__device__ inline void mengerFold(float4& z) {
    float a = fminf(z.x - z.y, 0.0f);
    z.x -= a;
    z.y += a;
    
    a = fminf(z.x - z.z, 0.0f);
    z.x -= a;
    z.z += a;
    
    a = fminf(z.y - z.z, 0.0f);
    z.y -= a;
    z.z += a;
}

__device__ inline void sierpinskiFold(float4& z) {
    if (z.x + z.y < 0.0f) {
        float temp = z.x;
        z.x = -z.y;
        z.y = -temp;
    }
    if (z.x + z.z < 0.0f) {
        float temp = z.x;
        z.x = -z.z;
        z.z = -temp;
    }
    if (z.y + z.z < 0.0f) {
        float temp = z.y;
        z.y = -z.z;
        z.z = -temp;
    }
}

__device__ inline void rotateY(float4& z, float angle) {
    float s = sinf(angle);
    float c = cosf(angle);
    float tempX = c * z.x - s * z.z;
    float tempZ = c * z.z + s * z.x;
    z.x = tempX;
    z.z = tempZ;
}

__device__ inline void rotateX(float4& z, float angle) {
    float s = sinf(angle);
    float c = cosf(angle);
    float tempY = c * z.y + s * z.z;
    float tempZ = c * z.z - s * z.y;
    z.y = tempY;
    z.z = tempZ;
}

// ==================== Distance Estimators ====================

__device__ float deMandelbox(float3 pos, const FractalParams& params) {
    float4 z = make_float4(pos.x, pos.y, pos.z, 1.0f);
    
    for (int i = 0; i < params.iterations; i++) {
        boxFold(z, params.foldRadius);
        sphereFold(z, params.minRadius, params.maxRadius);
        
        z.x = z.x * params.scale + params.offset.x;
        z.y = z.y * params.scale + params.offset.y;
        z.z = z.z * params.scale + params.offset.z;
        z.w = z.w * fabsf(params.scale);
    }
    
    float r = sqrtf(z.x * z.x + z.y * z.y + z.z * z.z);
    return (r - 2.0f) / fabsf(z.w);
}

__device__ float deMenger(float3 pos, const FractalParams& params) {
    float4 z = make_float4(pos.x, pos.y, pos.z, 1.0f);
    
    for (int i = 0; i < params.iterations; i++) {
        absFold(z);
        mengerFold(z);
        
        z.x = z.x * 3.0f - 2.0f;
        z.y = z.y * 3.0f - 2.0f;
        z.z = z.z * 3.0f - 2.0f;
        z.w = z.w * 3.0f;
        
        if (i < params.iterations - 1) {
            if (z.x < -1.0f) z.x += 2.0f;
            if (z.y < -1.0f) z.y += 2.0f;
            if (z.z < -1.0f) z.z += 2.0f;
        }
    }
    
    float r = sqrtf(z.x * z.x + z.y * z.y + z.z * z.z);
    return (r - 1.5f) / fabsf(z.w);
}

__device__ float deSierpinski(float3 pos, const FractalParams& params) {
    float4 z = make_float4(pos.x, pos.y, pos.z, 1.0f);
    
    for (int i = 0; i < params.iterations; i++) {
        sierpinskiFold(z);
        
        z.x = z.x * 2.0f - 1.0f;
        z.y = z.y * 2.0f - 1.0f;
        z.z = z.z * 2.0f - 1.0f;
        z.w = z.w * 2.0f;
    }
    
    float r = sqrtf(z.x * z.x + z.y * z.y + z.z * z.z);
    return (r - 1.0f) / fabsf(z.w);
}

__device__ float deTreePlanet(float3 pos, const FractalParams& params) {
    float4 z = make_float4(pos.x, pos.y, pos.z, 1.0f);
    
    for (int i = 0; i < params.iterations; i++) {
        rotateY(z, 0.44f);
        absFold(z);
        mengerFold(z);
        
        z.x = z.x * 1.3f - 2.0f;
        z.y = z.y * 1.3f - 4.8f;
        z.z = z.z * 1.3f;
        z.w = z.w * 1.3f;
        
        // Plane fold
        if (z.z < 0.0f) {
            z.z = -z.z;
        }
    }
    
    float r = sqrtf(z.x * z.x + z.y * z.y + z.z * z.z);
    return (r - 4.8f) / fabsf(z.w);
}

__device__ float sceneSDF(float3 pos, const FractalParams& params) {
    switch (params.type) {
        case 0: return deMandelbox(pos, params);
        case 1: return deMenger(pos, params);
        case 2: return deSierpinski(pos, params);
        case 3: return deTreePlanet(pos, params);
        default: return deMandelbox(pos, params);
    }
}

// Calculate normal using tetrahedron technique
__device__ float3 calcNormal(float3 pos, const FractalParams& params, float eps) {
    const float3 k1 = make_float3(1, -1, -1);
    const float3 k2 = make_float3(-1, -1, 1);
    const float3 k3 = make_float3(-1, 1, -1);
    const float3 k4 = make_float3(1, 1, 1);
    
    float3 n = make_float3(
        k1.x * sceneSDF(pos + k1 * eps, params) +
        k2.x * sceneSDF(pos + k2 * eps, params) +
        k3.x * sceneSDF(pos + k3 * eps, params) +
        k4.x * sceneSDF(pos + k4 * eps, params),
        
        k1.y * sceneSDF(pos + k1 * eps, params) +
        k2.y * sceneSDF(pos + k2 * eps, params) +
        k3.y * sceneSDF(pos + k3 * eps, params) +
        k4.y * sceneSDF(pos + k4 * eps, params),
        
        k1.z * sceneSDF(pos + k1 * eps, params) +
        k2.z * sceneSDF(pos + k2 * eps, params) +
        k3.z * sceneSDF(pos + k3 * eps, params) +
        k4.z * sceneSDF(pos + k4 * eps, params)
    );
    
    return normalize(n);
}

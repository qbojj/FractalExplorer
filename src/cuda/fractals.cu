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

__device__ inline void planeFold(float4& z, const float3& n, float d) {
    float dotVal = z.x * n.x + z.y * n.y + z.z * n.z - d;
    float k = fminf(0.0f, dotVal);
    // Reflect across plane when on the negative side of (dot - d)
    z.x -= 2.0f * k * n.x;
    z.y -= 2.0f * k * n.y;
    z.z -= 2.0f * k * n.z;
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
    const float boxSize = 1.0f;
    const float minR = 0.5f;
    const float maxR = 1.0f;
    const float scale = 2.0f;
    
    for (int i = 0; i < params.iterations; i++) {
        // FoldBox(1.0)
        boxFold(z, make_float3(boxSize, boxSize, boxSize));
        
        // FoldSphere(0.5, 1.0)
        sphereFold(z, minR, maxR);
        
        // FoldScaleOrigin(2.0) - scale around origin
        z.x = z.x * scale;
        z.y = z.y * scale;
        z.z = z.z * scale;
        z.w = z.w * scale;
    }
    
    float r = sqrtf(z.x * z.x + z.y * z.y + z.z * z.z);
    return (r - 2.0f) / fabsf(z.w);
}

__device__ float deMenger(float3 pos, const FractalParams& params) {
    float4 z = make_float4(pos.x, pos.y, pos.z, 1.0f);
    const float scale = 3.0f;
    const float3 translate = make_float3(-2.0f, -2.0f, 0.0f);
    
    for (int i = 0; i < params.iterations; i++) {
        // FoldAbs()
        absFold(z);
        
        // FoldMenger()
        mengerFold(z);
        
        // FoldScaleTranslate(3.0, (-2,-2,0))
        z.x = z.x * scale + translate.x;
        z.y = z.y * scale + translate.y;
        z.z = z.z * scale + translate.z;
        z.w = z.w * scale;
        
        // FoldPlane((0,0,-1), -1)
        planeFold(z, make_float3(0.0f, 0.0f, -1.0f), -1.0f);
    }
    
    float r = sqrtf(z.x * z.x + z.y * z.y + z.z * z.z);
    return (r - 2.0f) / fabsf(z.w);
}

__device__ float deSierpinski(float3 pos, const FractalParams& params) {
    float4 z = make_float4(pos.x, pos.y, pos.z, 1.0f);
    const float scale = 2.0f;
    const float offset = -1.0f;
    
    for (int i = 0; i < params.iterations; i++) {
        // FoldSierpinski()
        sierpinskiFold(z);
        
        // FoldScaleTranslate(2, -1)
        z.x = z.x * scale + offset;
        z.y = z.y * scale + offset;
        z.z = z.z * scale + offset;
        z.w = z.w * scale;
    }
    
    float r = sqrtf(z.x * z.x + z.y * z.y + z.z * z.z);
    return (r - 1.0f) / fabsf(z.w);
}

__device__ float deTreePlanet(float3 pos, const FractalParams& params) {
    float4 z = make_float4(pos.x, pos.y, pos.z, 1.0f);
    const float rot = params.rotationAngle;
    const float s = params.scale;
    const float3 t = params.offset; // use offset as translate
    
    for (int i = 0; i < params.iterations; i++) {
        // FoldRotateY(rot)
        rotateY(z, rot);

        // FoldAbs()
        absFold(z);

        // FoldMenger()
        mengerFold(z);

        // FoldScaleTranslate(scale, offset)
        z.x = z.x * s + t.x;
        z.y = z.y * s + t.y;
        z.z = z.z * s + t.z;
        z.w = z.w * fabsf(s);

        // FoldPlane((0,0,-1), 0) matches PySpace: reflect when dot(z,n) - d < 0
        planeFold(z, make_float3(0.0f, 0.0f, -1.0f), 0.0f);
    }
    
    // Box(4.8) distance estimator
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

// Orbit coloring approximations (PySpace-inspired)
__device__ float3 orbitColor(float3 pos, const FractalParams& params) {
    float3 orbit = make_float3(1e20f, 1e20f, 1e20f);

    if (params.type == 0) { // Mandelbox: PySpace-aligned
        float4 z = make_float4(pos.x, pos.y, pos.z, 1.0f);
        const float boxSize = 1.0f;
        const float minR = 0.5f;
        const float maxR = 1.0f;
        const float scale = 2.0f;
        
        for (int i = 0; i < params.iterations; i++) {
            boxFold(z, make_float3(boxSize, boxSize, boxSize));
            sphereFold(z, minR, maxR);
            // Track minimum absolute values
            orbit.x = fminf(orbit.x, fabsf(z.x));
            orbit.y = fminf(orbit.y, fabsf(z.y));
            orbit.z = fminf(orbit.z, fabsf(z.z));
            // Scale
            z.x = z.x * scale;
            z.y = z.y * scale;
            z.z = z.z * scale;
            z.w = z.w * scale;
        }
    } else if (params.type == 1) { // Menger: PySpace-aligned
        float4 z = make_float4(pos.x, pos.y, pos.z, 1.0f);
        const float scale = 3.0f;
        const float3 translate = make_float3(-2.0f, -2.0f, 0.0f);
        
        for (int i = 0; i < params.iterations; i++) {
            absFold(z);
            mengerFold(z);
            // Track minimum absolute values
            orbit.x = fminf(orbit.x, fabsf(z.x));
            orbit.y = fminf(orbit.y, fabsf(z.y));
            orbit.z = fminf(orbit.z, fabsf(z.z));
            z.x = z.x * scale + translate.x;
            z.y = z.y * scale + translate.y;
            z.z = z.z * scale + translate.z;
            z.w = z.w * scale;
            planeFold(z, make_float3(0.0f, 0.0f, -1.0f), -1.0f);
        }
    } else if (params.type == 2) { // Sierpinski: PySpace-aligned
        float4 z = make_float4(pos.x, pos.y, pos.z, 1.0f);
        const float scale = 2.0f;
        const float offset = -1.0f;
        
        for (int i = 0; i < params.iterations; i++) {
            sierpinskiFold(z);
            // Track minimum absolute values
            orbit.x = fminf(orbit.x, fabsf(z.x));
            orbit.y = fminf(orbit.y, fabsf(z.y));
            orbit.z = fminf(orbit.z, fabsf(z.z));
            z.x = z.x * scale + offset;
            z.y = z.y * scale + offset;
            z.z = z.z * scale + offset;
            z.w = z.w * scale;
        }
    } else if (params.type == 3) { // Tree Planet: PySpace-aligned
        float4 z = make_float4(pos.x, pos.y, pos.z, 1.0f);
        const float3 orbitScale = make_float3(0.24f, 2.28f, 7.6f);
        const float rot = params.rotationAngle;
        const float s = params.scale;
        const float3 t = params.offset;
        
        for (int i = 0; i < params.iterations; i++) {
            rotateY(z, rot);
            absFold(z);
            mengerFold(z);
            // OrbitMinAbs with scaling
            orbit.x = fminf(orbit.x, fabsf(z.x * orbitScale.x));
            orbit.y = fminf(orbit.y, fabsf(z.y * orbitScale.y));
            orbit.z = fminf(orbit.z, fabsf(z.z * orbitScale.z));
            z.x = z.x * s + t.x;
            z.y = z.y * s + t.y;
            z.z = z.z * s + t.z;
            z.w = z.w * fabsf(s);
            planeFold(z, make_float3(0.0f, 0.0f, -1.0f), 0.0f);
        }
    }

    // Normalize orbit to 0..1 range for coloring
    float3 col = make_float3(0.0f, 0.0f, 0.0f);
    col.x = clamp(orbit.x * 0.5f, 0.0f, 1.0f);
    col.y = clamp(orbit.y * 0.2f, 0.0f, 1.0f);
    col.z = clamp(orbit.z * 0.1f, 0.0f, 1.0f);
    return col;
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

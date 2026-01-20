#include "../include/types.h"

// Forward declare fractal functions
__device__ float sceneSDF(float3 pos, const FractalParams& params);
__device__ float3 calcNormal(float3 pos, const FractalParams& params, float eps);
__device__ float3 orbitColor(float3 pos, const FractalParams& params);

// ==================== Ray Marching ====================

__device__ float4 rayMarch(float3 origin, float3 direction, const FractalParams& fractalParams, const RenderParams& renderParams) {
    float t = 0.0f;
    float minDist = 1e10f;
    
    for (int i = 0; i < renderParams.maxSteps; i++) {
        float3 pos = origin + direction * t;
        float dist = sceneSDF(pos, fractalParams);
        
        minDist = fminf(minDist, dist);
        
        if (dist < renderParams.epsilon) {
            // Hit! Return distance, step count, and min distance
            return make_float4(t, (float)i, t, minDist);
        }
        
        t += dist;
        
        if (t > renderParams.maxDist) {
            break;
        }
    }
    
    // Miss
    return make_float4(-1.0f, (float)renderParams.maxSteps, t, minDist);
}

// ==================== Shading ====================

__device__ float3 shade(float3 pos, float3 rayDir, const FractalParams& fractalParams, const RenderParams& renderParams) {
    float3 normal = calcNormal(pos, fractalParams, renderParams.epsilon * 10.0f);
    
    // Orbit-based color (PySpace-inspired)
    float3 baseColor = orbitColor(pos, fractalParams);
    // Boost a little to avoid dark scenes
    baseColor = clamp(baseColor + make_float3(0.1f, 0.1f, 0.1f), 0.0f, 1.0f);
    
    // Ambient
    float3 ambient = baseColor * renderParams.ambientStrength;
    
    // Diffuse
    float diff = fmaxf(dot(normal, renderParams.lightDir), 0.0f);
    float3 diffuse = baseColor * diff * renderParams.diffuseStrength;
    
    // Specular
    float3 reflected = reflect(-renderParams.lightDir, normal);
    float spec = powf(fmaxf(dot(reflected, -rayDir), 0.0f), (float)renderParams.specularPower);
    float3 specular = make_float3(1.0f, 1.0f, 1.0f) * spec * renderParams.specularStrength;
    
    // Simple shadow (optional - can be disabled for performance)
    float shadow = 1.0f;
    float3 shadowOrigin = pos + normal * renderParams.epsilon * 20.0f;
    float4 shadowResult = rayMarch(shadowOrigin, renderParams.lightDir, fractalParams, renderParams);
    if (shadowResult.x > 0.0f) {
        shadow = 0.3f;
    }
    
    return ambient + (diffuse + specular) * shadow;
}

// ==================== Main Kernel ====================

__global__ void renderKernel(
    uchar3* output,
    const CameraParams camera,
    const FractalParams fractalParams,
    const RenderParams renderParams
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (px >= camera.width || py >= camera.height) return;
    
    // Generate ray
    float aspectRatio = (float)camera.width / (float)camera.height;
    float fovRadians = camera.fov * 3.14159265f / 180.0f;
    float focalDist = 1.0f / tanf(fovRadians * 0.5f);
    
    float u = (2.0f * (px + 0.5f) / camera.width - 1.0f) * aspectRatio;
    float v = -(2.0f * (py + 0.5f) / camera.height - 1.0f);
    
    // Ray in camera space
    float4 rayCamera = make_float4(u, v, -focalDist, 0.0f);
    float rayLen = sqrtf(rayCamera.x * rayCamera.x + rayCamera.y * rayCamera.y + rayCamera.z * rayCamera.z);
    rayCamera.x /= rayLen;
    rayCamera.y /= rayLen;
    rayCamera.z /= rayLen;
    
    // Transform to world space
    float4 rayWorld = camera.transform * rayCamera;
    float3 rayDir = normalize(make_float3(rayWorld.x, rayWorld.y, rayWorld.z));
    
    float4 originWorld = camera.transform * make_float4(0, 0, 0, 1);
    float3 origin = make_float3(originWorld.x, originWorld.y, originWorld.z);
    
    // Ray march
    float4 result = rayMarch(origin, rayDir, fractalParams, renderParams);
    
    float3 color;
    
    if (result.x > 0.0f) {
        // Hit - shade the surface
        float3 hitPos = origin + rayDir * result.x;
        color = shade(hitPos, rayDir, fractalParams, renderParams);
        
        // Add some glow effect based on minimum distance encountered
        float glow = expf(-result.w * 20.0f);
        color = color + make_float3(0.1f, 0.15f, 0.2f) * glow * 0.5f;
    } else {
        // Miss - background with glow
        color = renderParams.backgroundColor;
        
        // Glow effect
        float glow = 1.0f - fminf(result.w * 5.0f, 1.0f);
        color = color + make_float3(0.1f, 0.15f, 0.2f) * glow * glow;
    }
    
    // Clamp and convert to unsigned byte
    color.x = clamp(color.x, 0.0f, 1.0f);
    color.y = clamp(color.y, 0.0f, 1.0f);
    color.z = clamp(color.z, 0.0f, 1.0f);
    
    int idx = py * camera.width + px;
    output[idx] = make_uchar3(
        (unsigned char)(color.x * 255),
        (unsigned char)(color.y * 255),
        (unsigned char)(color.z * 255)
    );
}

#include "../include/window.h"
#include "../include/types.h"
#include "../include/raymarcher.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>

class FractalExplorer {
public:
    FractalExplorer(int width, int height);
    ~FractalExplorer();
    
    void run();
    
private:
    void initCUDA();
    void render();
    void switchFractal(int type);
    void updateInterpolation(float deltaTime);
    void startMorphTo(int targetType);
    
    Window window;
    int width, height;
    
    // CUDA resources
    uchar3* d_output;  // Device framebuffer
    uchar3* h_output;  // Host framebuffer
    
    // Rendering parameters
    FractalParams fractalParams;
    RenderParams renderParams;
    
    int currentFractalType = 0;
    double lastTime = 0;
    int frameCount = 0;
    
    // Interpolation state
    bool isMorphing = false;
    float morphSpeed = 0.5f;  // Units per second
    FractalParams sourceParams;
    FractalParams targetParams;
    
    // Auto-cycle mode
    bool autoCycle = false;
    float cycleTimer = 0.0f;
    float cycleInterval = 5.0f;  // Seconds between auto-switches
};

FractalExplorer::FractalExplorer(int w, int h)
    : window(w, h, "Fractal Explorer - CUDA Ray Marcher"), width(w), height(h) {
    
    initCUDA();
    
    // Initialize fractal parameters for Mandelbox
    fractalParams.type = 0;
    fractalParams.iterations = 12;
    fractalParams.scale = 2.5f;
    fractalParams.minRadius = 0.25f;
    fractalParams.maxRadius = 1.0f;
    fractalParams.foldRadius = make_float3(1.0f, 1.0f, 1.0f);
    fractalParams.offset = make_float3(0.0f, 0.0f, 0.0f);
    fractalParams.rotationAngle = 0.0f;
    fractalParams.colorOrbit = make_float3(1.0f, 1.0f, 1.0f);
    fractalParams.morphFactor = 0.0f;
    fractalParams.targetType = 0;
    fractalParams.targetScale = 2.5f;
    fractalParams.targetIterations = 12;
    
    // Initialize rendering parameters
    renderParams.maxSteps = 512;
    renderParams.maxDist = 50.0f;
    renderParams.epsilon = 1e-3f;
    renderParams.lightDir = normalize(make_float3(0.5f, 1.0f, 0.3f));
    renderParams.backgroundColor = make_float3(0.05f, 0.05f, 0.1f);
    renderParams.ambientStrength = 0.2f;
    renderParams.diffuseStrength = 0.7f;
    renderParams.specularStrength = 0.5f;
    renderParams.specularPower = 32;
    
    std::cout << "Fractal Explorer initialized" << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  WASD - Move camera" << std::endl;
    std::cout << "  Space/Shift - Move up/down" << std::endl;
    std::cout << "  Mouse - Look around" << std::endl;
    std::cout << "  ESC - Exit" << std::endl;
}

FractalExplorer::~FractalExplorer() {
    cudaFree(d_output);
    delete[] h_output;
}

void FractalExplorer::initCUDA() {
    // Allocate device memory
    size_t imageSize = width * height * sizeof(uchar3);
    cudaMalloc(&d_output, imageSize);
    
    // Allocate host memory
    h_output = new uchar3[width * height];
    
    std::cout << "CUDA initialized" << std::endl;
    std::cout << "Image size: " << width << "x" << height << std::endl;
}

void FractalExplorer::render() {
    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    CameraParams camParams = window.getCamera().getParams();
    
    // Call kernel
    renderKernel<<<gridSize, blockSize>>>(d_output, camParams, fractalParams, renderParams);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Wait for kernel to finish
    cudaDeviceSynchronize();
    
    // Copy result to host
    size_t imageSize = width * height * sizeof(uchar3);
    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);
    
    // Display
    window.display(h_output);
}

void FractalExplorer::switchFractal(int type) {
    if (type == currentFractalType) return;
    
    currentFractalType = type;
    fractalParams.type = type;
    isMorphing = false;
    
    // Set base parameters for this fractal type
    switch (type) {
        case 0: // Mandelbox
            fractalParams.iterations = 12;
            fractalParams.scale = 2.5f;
            fractalParams.minRadius = 0.25f;
            fractalParams.maxRadius = 1.0f;
            fractalParams.foldRadius = make_float3(1.0f, 1.0f, 1.0f);
            std::cout << "Switched to Mandelbox" << std::endl;
            break;
            
        case 1: // Menger
            fractalParams.iterations = 5;
            fractalParams.scale = 3.0f;
            std::cout << "Switched to Menger Sponge" << std::endl;
            break;
            
        case 2: // Sierpinski
            fractalParams.iterations = 9;
            fractalParams.scale = 2.4f;
            std::cout << "Switched to Sierpinski Tetrahedron" << std::endl;
            break;
            
        case 3: // Tree Planet
            fractalParams.iterations = 30;
            fractalParams.scale = 1.3f;
            fractalParams.offset = make_float3(-2.0f, -4.8f, 0.0f);
            fractalParams.rotationAngle = 0.44f;
            std::cout << "Switched to Tree Planet" << std::endl;
            break;
    }
    
    // Reset morph to start new parameter exploration
    fractalParams.morphFactor = 0.0f;
    startMorphTo(type);
}

void FractalExplorer::startMorphTo(int targetType) {
    // Save current state as source
    sourceParams = fractalParams;
    sourceParams.morphFactor = 0.0f;
    
    // Set target parameters - vary parameters within the same fractal type
    targetParams = fractalParams;
    targetParams.morphFactor = 1.0f;
    
    switch (targetType) {
        case 0: // Mandelbox - vary scale and radii
            targetParams.iterations = 12;
            targetParams.scale = 3.0f + sinf((float)window.getTime() * 0.3f) * 0.5f;
            targetParams.minRadius = 0.15f + cosf((float)window.getTime() * 0.4f) * 0.1f;
            targetParams.maxRadius = 0.8f + sinf((float)window.getTime() * 0.5f) * 0.3f;
            targetParams.foldRadius = make_float3(
                0.8f + sinf((float)window.getTime() * 0.2f) * 0.3f,
                0.8f + cosf((float)window.getTime() * 0.25f) * 0.3f,
                0.8f + sinf((float)window.getTime() * 0.3f) * 0.3f
            );
            break;
            
        case 1: // Menger - vary scale/iterations for spacing
            targetParams.iterations = 4 + (int)(sinf((float)window.getTime() * 0.18f) * 2.0f);
            targetParams.scale = 2.6f + cosf((float)window.getTime() * 0.22f) * 0.6f;
            break;
            
        case 2: // Sierpinski - vary iterations and scale for air gaps
            targetParams.iterations = 7 + (int)(sinf((float)window.getTime() * 0.3f) * 4.0f);
            targetParams.scale = 2.2f + cosf((float)window.getTime() * 0.32f) * 0.45f;
            break;
            
        case 3: // Tree Planet - closer to PySpace recipe
            targetParams.iterations = 24 + (int)(sinf((float)window.getTime() * 0.22f) * 6.0f);
            targetParams.scale = 1.25f + cosf((float)window.getTime() * 0.18f) * 0.15f;
            targetParams.rotationAngle = 0.44f + sinf((float)window.getTime() * 0.35f) * 0.05f;
            targetParams.offset = make_float3(
                -2.0f + cosf((float)window.getTime() * 0.17f) * 0.2f,
                -4.8f + sinf((float)window.getTime() * 0.19f) * 0.4f,
                0.0f
            );
            break;
    }
    
    isMorphing = true;
    fractalParams.morphFactor = 0.0f;
    fractalParams.targetType = targetType;
    fractalParams.targetScale = targetParams.scale;
    fractalParams.targetIterations = targetParams.iterations;
}

void FractalExplorer::updateInterpolation(float deltaTime) {
    // Auto-cycle through parameter variations within current fractal
    if (autoCycle && !isMorphing) {
        cycleTimer += deltaTime;
        if (cycleTimer >= cycleInterval) {
            cycleTimer = 0.0f;
            // Start new parameter variation within same fractal type
            startMorphTo(currentFractalType);
        }
    }
    
    if (!isMorphing) return;
    
    // Update morph factor
    fractalParams.morphFactor += morphSpeed * deltaTime;
    
    if (fractalParams.morphFactor >= 1.0f) {
        // Morphing complete - restart with new target
        fractalParams.morphFactor = 1.0f;
        // Copy target parameters but keep the type unchanged
        fractalParams.scale = targetParams.scale;
        fractalParams.minRadius = targetParams.minRadius;
        fractalParams.maxRadius = targetParams.maxRadius;
        fractalParams.foldRadius = targetParams.foldRadius;
        fractalParams.offset = targetParams.offset;
        fractalParams.rotationAngle = targetParams.rotationAngle;
        fractalParams.iterations = targetParams.iterations;
        fractalParams.morphFactor = 1.0f;
        isMorphing = false;
        
        // If auto-cycle is on, immediately start morphing to new parameters
        if (autoCycle) {
            cycleTimer = 0.0f;
            startMorphTo(currentFractalType);
        }
    } else {
        // Interpolate parameters
        float t = fractalParams.morphFactor;
        
        fractalParams.scale = sourceParams.scale * (1.0f - t) + targetParams.scale * t;
        fractalParams.minRadius = sourceParams.minRadius * (1.0f - t) + targetParams.minRadius * t;
        fractalParams.maxRadius = sourceParams.maxRadius * (1.0f - t) + targetParams.maxRadius * t;
        
        // Interpolate float3 parameters
        fractalParams.foldRadius.x = sourceParams.foldRadius.x * (1.0f - t) + targetParams.foldRadius.x * t;
        fractalParams.foldRadius.y = sourceParams.foldRadius.y * (1.0f - t) + targetParams.foldRadius.y * t;
        fractalParams.foldRadius.z = sourceParams.foldRadius.z * (1.0f - t) + targetParams.foldRadius.z * t;
        
        fractalParams.offset.x = sourceParams.offset.x * (1.0f - t) + targetParams.offset.x * t;
        fractalParams.offset.y = sourceParams.offset.y * (1.0f - t) + targetParams.offset.y * t;
        fractalParams.offset.z = sourceParams.offset.z * (1.0f - t) + targetParams.offset.z * t;
        
        fractalParams.rotationAngle = sourceParams.rotationAngle * (1.0f - t) + targetParams.rotationAngle * t;
    }
}

void FractalExplorer::run() {
    lastTime = window.getTime();
    
    while (!window.shouldClose()) {
        // Calculate delta time
        double currentTime = window.getTime();
        float deltaTime = static_cast<float>(currentTime - lastTime);
        lastTime = currentTime;
        
        // Update camera
        window.getCamera().update(deltaTime);
        
        // Update fractal interpolation
        updateInterpolation(deltaTime);
        
        // Poll events
        window.pollEvents();
        
        // Fractal switching
        static bool key1Pressed = false, key2Pressed = false, key3Pressed = false, key4Pressed = false;
        if (window.isKeyPressed(GLFW_KEY_1) && !key1Pressed) { switchFractal(0); key1Pressed = true; }
        else if (!window.isKeyPressed(GLFW_KEY_1)) key1Pressed = false;
        
        if (window.isKeyPressed(GLFW_KEY_2) && !key2Pressed) { switchFractal(1); key2Pressed = true; }
        else if (!window.isKeyPressed(GLFW_KEY_2)) key2Pressed = false;
        
        if (window.isKeyPressed(GLFW_KEY_3) && !key3Pressed) { switchFractal(2); key3Pressed = true; }
        else if (!window.isKeyPressed(GLFW_KEY_3)) key3Pressed = false;
        
        if (window.isKeyPressed(GLFW_KEY_4) && !key4Pressed) { switchFractal(3); key4Pressed = true; }
        else if (!window.isKeyPressed(GLFW_KEY_4)) key4Pressed = false;
        
        // Morph speed controls
        {
            float prevSpeed = morphSpeed;
            if (window.isKeyPressed(GLFW_KEY_EQUAL)) {
                morphSpeed += 1.0f * deltaTime;
                if (morphSpeed > 5.0f) morphSpeed = 5.0f;
            }
            if (window.isKeyPressed(GLFW_KEY_MINUS)) {
                morphSpeed -= 1.0f * deltaTime;
                if (morphSpeed < 0.1f) morphSpeed = 0.1f;
            }
            if (fabsf(prevSpeed - morphSpeed) > 1e-4f) {
                std::cout << "Morph speed: " << morphSpeed << std::endl;
                // If not currently morphing, start a new parameter morph to reflect the new speed
                if (!isMorphing) {
                    startMorphTo(currentFractalType);
                }
            }
        }
        
        // Toggle auto-cycle
        static bool tabPressed = false;
        if (window.isKeyPressed(GLFW_KEY_TAB) && !tabPressed) {
            autoCycle = !autoCycle;
            cycleTimer = 0.0f;
            std::cout << "Auto-cycle: " << (autoCycle ? "ON" : "OFF") << " | Morph speed: " << morphSpeed << std::endl;
            tabPressed = true;
        } else if (!window.isKeyPressed(GLFW_KEY_TAB)) {
            tabPressed = false;
        }
        
        // Render
        render();
        
        // FPS counter
        frameCount++;
        if (frameCount % 60 == 0) {
            std::cout << "FPS: " << (int)(1.0 / deltaTime) << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    int width = 1280;
    int height = 720;
    
    if (argc >= 3) {
        width = std::atoi(argv[1]);
        height = std::atoi(argv[2]);
    }
    
    std::cout << "Starting Fractal Explorer..." << std::endl;
    
    try {
        FractalExplorer explorer(width, height);
        explorer.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "Exiting..." << std::endl;
    return 0;
}

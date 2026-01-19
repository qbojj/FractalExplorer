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
    fractalParams.iterations = 15;
    fractalParams.scale = 2.0f;
    fractalParams.minRadius = 0.5f;
    fractalParams.maxRadius = 1.0f;
    fractalParams.foldRadius = make_float3(1.0f, 1.0f, 1.0f);
    fractalParams.offset = make_float3(0.0f, 0.0f, 0.0f);
    fractalParams.rotationAngle = 0.0f;
    fractalParams.colorOrbit = make_float3(1.0f, 1.0f, 1.0f);
    fractalParams.morphFactor = 0.0f;
    fractalParams.targetType = 0;
    fractalParams.targetScale = 2.0f;
    fractalParams.targetIterations = 15;
    
    // Initialize rendering parameters
    renderParams.maxSteps = 128;
    renderParams.maxDist = 50.0f;
    renderParams.epsilon = 0.001f;
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
    startMorphTo(type);
}

void FractalExplorer::startMorphTo(int targetType) {
    // Save current state as source
    sourceParams = fractalParams;
    sourceParams.morphFactor = 0.0f;
    
    // Set target parameters
    targetParams = fractalParams;
    targetParams.type = targetType;
    targetParams.morphFactor = 1.0f;
    
    switch (targetType) {
        case 0: // Mandelbox
            targetParams.iterations = 15;
            targetParams.scale = 2.0f;
            targetParams.minRadius = 0.5f;
            targetParams.maxRadius = 1.0f;
            targetParams.foldRadius = make_float3(1.0f, 1.0f, 1.0f);
            std::cout << "Morphing to Mandelbox" << std::endl;
            break;
            
        case 1: // Menger
            targetParams.iterations = 8;
            targetParams.scale = 3.0f;
            std::cout << "Morphing to Menger Sponge" << std::endl;
            break;
            
        case 2: // Sierpinski
            targetParams.iterations = 10;
            targetParams.scale = 2.0f;
            std::cout << "Morphing to Sierpinski Tetrahedron" << std::endl;
            break;
            
        case 3: // Tree Planet
            targetParams.iterations = 30;
            targetParams.scale = 1.3f;
            std::cout << "Morphing to Tree Planet" << std::endl;
            break;
    }
    
    isMorphing = true;
    fractalParams.morphFactor = 0.0f;
    fractalParams.targetType = targetType;
    fractalParams.targetScale = targetParams.scale;
    fractalParams.targetIterations = targetParams.iterations;
}

void FractalExplorer::updateInterpolation(float deltaTime) {
    // Auto-cycle through fractals
    if (autoCycle && !isMorphing) {
        cycleTimer += deltaTime;
        if (cycleTimer >= cycleInterval) {
            cycleTimer = 0.0f;
            int nextFractal = (currentFractalType + 1) % 4;
            switchFractal(nextFractal);
        }
    }
    
    if (!isMorphing) return;
    
    // Update morph factor
    fractalParams.morphFactor += morphSpeed * deltaTime;
    
    if (fractalParams.morphFactor >= 1.0f) {
        // Morphing complete
        fractalParams.morphFactor = 1.0f;
        fractalParams = targetParams;
        fractalParams.type = currentFractalType;
        isMorphing = false;
        std::cout << "Morph complete" << std::endl;
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
        if (window.isKeyPressed(GLFW_KEY_EQUAL)) {
            morphSpeed += 1.0f * deltaTime;
            if (morphSpeed > 5.0f) morphSpeed = 5.0f;
        }
        if (window.isKeyPressed(GLFW_KEY_MINUS)) {
            morphSpeed -= 1.0f * deltaTime;
            if (morphSpeed < 0.1f) morphSpeed = 0.1f;
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

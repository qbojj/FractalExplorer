#include "../include/window.h"
#include "../include/types.h"
#include "../include/raymarcher.h"
#include "../include/cpu_raymarcher.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>

class FractalExplorer {
public:
    FractalExplorer(int width, int height, bool useCPU = false);
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
    bool useCPU;  // Flag to use CPU or GPU rendering
    
    // CUDA resources
    uchar3* d_output;  // Device framebuffer
    uchar3* h_output;  // Host framebuffer
    
    // CPU rendering
    CPURayMarcher* cpuRayMarcher;
    
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

FractalExplorer::FractalExplorer(int w, int h, bool useCPU)
    : window(w, h, useCPU ? "Fractal Explorer - CPU Ray Marcher" : "Fractal Explorer - CUDA Ray Marcher"), 
      width(w), height(h), useCPU(useCPU), cpuRayMarcher(nullptr) {
    
    if (!useCPU) {
        initCUDA();
    } else {
        h_output = new uchar3[w * h];
        cpuRayMarcher = new CPURayMarcher(w, h);
        std::cout << "CPU ray marcher initialized" << std::endl;
        std::cout << "Image size: " << w << "x" << h << std::endl;
    }
    
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
    renderParams.maxSteps = 1000;
    renderParams.maxDist = 50.0f;
    renderParams.minDist = 0.00001f;
    renderParams.lodMultiplier = 10.0f;
    renderParams.glowSharpness = 8.0f;
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
    if (!useCPU) {
        cudaFree(d_output);
    }
    if (cpuRayMarcher) {
        delete cpuRayMarcher;
    }
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
    if (useCPU) {
        // Render using CPU
        cpuRayMarcher->render(h_output, window.getCamera().getParams(), fractalParams, renderParams);
    } else {
        // Render using CUDA
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
    }
    
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
        case 0: // Mandelbox: FoldBox(1.0) + FoldSphere(0.5,1.0) + FoldScaleOrigin(2.0), 16 iterations
            fractalParams.iterations = 16;
            fractalParams.scale = 2.0f;  // Not used in hardcoded DE but kept for consistency
            fractalParams.minRadius = 0.5f;
            fractalParams.maxRadius = 1.0f;
            fractalParams.foldRadius = make_float3(1.0f, 1.0f, 1.0f);
            fractalParams.offset = make_float3(0.0f, 0.0f, 0.0f);
            fractalParams.rotationAngle = 0.0f;
            std::cout << "Switched to Mandelbox" << std::endl;
            break;
            
        case 1: // Menger: FoldAbs + FoldMenger + FoldScaleTranslate(3.0,-2,-2,0) + FoldPlane, 8 iterations
            fractalParams.iterations = 8;
            fractalParams.scale = 3.0f;  // Used in hardcoded DE
            fractalParams.offset = make_float3(-2.0f, -2.0f, 0.0f);
            fractalParams.rotationAngle = 0.0f;
            std::cout << "Switched to Menger Sponge" << std::endl;
            break;
            
        case 2: // Sierpinski: FoldSierpinski + FoldScaleTranslate(2, -1), 9 iterations
            fractalParams.iterations = 9;
            fractalParams.scale = 2.0f;  // Used in hardcoded DE
            fractalParams.offset = make_float3(-1.0f, -1.0f, -1.0f);
            fractalParams.rotationAngle = 0.0f;
            std::cout << "Switched to Sierpinski Tetrahedron" << std::endl;
            break;
            
        case 3: // Tree Planet
            fractalParams.iterations = 30;
            fractalParams.scale = 1.3f;
            fractalParams.offset = make_float3(-2.0f, -4.8f, 0.0f);
            fractalParams.rotationAngle = 0.44f;
            std::cout << "Switched to Tree Planet" << std::endl;
            break;
            
        case 4: // Butterweed Hills: OrbitSum coloring with rotation
            fractalParams.iterations = 30;
            fractalParams.scale = 1.5f;
            fractalParams.offset = make_float3(-1.0f, -0.5f, -0.2f);
            fractalParams.rotationAngle = 0.0f;
            std::cout << "Switched to Butterweed Hills" << std::endl;
            break;
            
        case 5: // Mausoleum: OrbitMax coloring
            fractalParams.iterations = 8;
            fractalParams.scale = 3.28f;
            fractalParams.offset = make_float3(-5.27f, -0.34f, 0.0f);
            fractalParams.rotationAngle = 0.0f;
            std::cout << "Switched to Mausoleum" << std::endl;
            break;
            
        case 6: // Snow Stadium: Complex rotational fractal
            fractalParams.iterations = 30;
            fractalParams.scale = 1.57f;
            fractalParams.offset = make_float3(-6.61f, -4.0f, -2.42f);
            fractalParams.rotationAngle = 0.0f;
            std::cout << "Switched to Snow Stadium" << std::endl;
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
    
    // Set target parameters - rich parameter space exploration within fractal type
    targetParams = fractalParams;
    targetParams.morphFactor = 1.0f;
    
    double t = window.getTime();
    
    switch (targetType) {
        case 0: { // Mandelbox - explore fold radius space and sphere radii
            targetParams.iterations = 16;
            // Vary minRadius and maxRadius for different fold characteristics
            float minRadTarget = 0.3f + sinf((float)t * 0.18f) * 0.2f;  // 0.1 to 0.5
            float maxRadTarget = 0.7f + cosf((float)t * 0.22f) * 0.3f;  // 0.4 to 1.0
            targetParams.minRadius = clamp(minRadTarget, 0.1f, 0.8f);
            targetParams.maxRadius = clamp(maxRadTarget, minRadTarget + 0.1f, 1.2f);
            
            // Vary fold radius asymmetrically
            targetParams.foldRadius = make_float3(
                1.0f + sinf((float)t * 0.19f) * 0.25f,
                1.0f + cosf((float)t * 0.24f) * 0.25f,
                1.0f + sinf((float)t * 0.31f) * 0.25f
            );
            break;
        }
            
        case 1: { // Menger - explore offset space and asymmetry
            targetParams.iterations = 8;
            targetParams.scale = 3.0f;
            
            // Create varied offset patterns
            float offsetAmp = 0.5f + sinf((float)t * 0.15f) * 0.2f;
            targetParams.offset = make_float3(
                -2.0f + sinf((float)t * 0.21f) * offsetAmp,
                -2.0f + cosf((float)t * 0.25f) * offsetAmp,
                sinf((float)t * 0.18f) * 0.4f
            );
            break;
        }
            
        case 2: { // Sierpinski - explore scale variations and offset drift
            targetParams.iterations = 9;
            
            // Smooth scale variation around base
            targetParams.scale = 2.0f + sinf((float)t * 0.16f) * 0.3f;
            
            // Asymmetric offset drifts
            float driftAmount = 0.4f + cosf((float)t * 0.12f) * 0.15f;
            targetParams.offset = make_float3(
                -1.0f + sinf((float)t * 0.19f) * driftAmount,
                -1.0f + cosf((float)t * 0.23f) * driftAmount,
                -1.0f + sinf((float)t * 0.27f) * driftAmount
            );
            break;
        }
            
        case 3: { // Tree Planet - full parameter space exploration
            targetParams.iterations = 30;
            
            // Scale breathing effect
            targetParams.scale = 1.3f + sinf((float)t * 0.15f) * 0.08f;
            
            // Rotation variation creates dramatic visual changes
            targetParams.rotationAngle = 0.44f + cosf((float)t * 0.28f) * 0.12f;
            
            // Offset creates positional drift with different frequency per axis
            float offsetX = cosf((float)t * 0.17f) * 0.3f;
            float offsetY = sinf((float)t * 0.19f) * 0.6f;
            targetParams.offset = make_float3(
                -2.0f + offsetX,
                -4.8f + offsetY,
                sinf((float)t * 0.14f) * 0.2f
            );
            break;
        }
            
        case 4: { // Butterweed Hills - scale and offset variations
            targetParams.iterations = 30;
            targetParams.scale = 1.5f + sinf((float)t * 0.12f) * 0.1f;
            targetParams.offset = make_float3(
                -1.0f + cosf((float)t * 0.16f) * 0.2f,
                -0.5f + sinf((float)t * 0.2f) * 0.15f,
                -0.2f + cosf((float)t * 0.14f) * 0.1f
            );
            break;
        }
            
        case 5: { // Mausoleum - box fold and offset modulation
            targetParams.iterations = 8;
            targetParams.scale = 3.28f + sinf((float)t * 0.1f) * 0.15f;
            targetParams.foldRadius = make_float3(
                0.34f + cosf((float)t * 0.18f) * 0.08f,
                0.34f + sinf((float)t * 0.22f) * 0.08f,
                0.34f + cosf((float)t * 0.26f) * 0.08f
            );
            targetParams.offset = make_float3(
                -5.27f + sinf((float)t * 0.13f) * 0.4f,
                -0.34f + cosf((float)t * 0.17f) * 0.2f,
                0.0f
            );
            break;
        }
            
        case 6: { // Snow Stadium - rotation and offset exploration
            targetParams.iterations = 30;
            targetParams.scale = 1.57f + cosf((float)t * 0.11f) * 0.05f;
            targetParams.offset = make_float3(
                -6.61f + sinf((float)t * 0.14f) * 0.5f,
                -4.0f + cosf((float)t * 0.18f) * 0.3f,
                -2.42f + sinf((float)t * 0.22f) * 0.3f
            );
            break;
        }
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
        static bool key5Pressed = false, key6Pressed = false, key7Pressed = false;
        
        if (window.isKeyPressed(GLFW_KEY_1) && !key1Pressed) { switchFractal(0); key1Pressed = true; }
        else if (!window.isKeyPressed(GLFW_KEY_1)) key1Pressed = false;
        
        if (window.isKeyPressed(GLFW_KEY_2) && !key2Pressed) { switchFractal(1); key2Pressed = true; }
        else if (!window.isKeyPressed(GLFW_KEY_2)) key2Pressed = false;
        
        if (window.isKeyPressed(GLFW_KEY_3) && !key3Pressed) { switchFractal(2); key3Pressed = true; }
        else if (!window.isKeyPressed(GLFW_KEY_3)) key3Pressed = false;
        
        if (window.isKeyPressed(GLFW_KEY_4) && !key4Pressed) { switchFractal(3); key4Pressed = true; }
        else if (!window.isKeyPressed(GLFW_KEY_4)) key4Pressed = false;
        
        if (window.isKeyPressed(GLFW_KEY_5) && !key5Pressed) { switchFractal(4); key5Pressed = true; }
        else if (!window.isKeyPressed(GLFW_KEY_5)) key5Pressed = false;
        
        if (window.isKeyPressed(GLFW_KEY_6) && !key6Pressed) { switchFractal(5); key6Pressed = true; }
        else if (!window.isKeyPressed(GLFW_KEY_6)) key6Pressed = false;
        
        if (window.isKeyPressed(GLFW_KEY_7) && !key7Pressed) { switchFractal(6); key7Pressed = true; }
        else if (!window.isKeyPressed(GLFW_KEY_7)) key7Pressed = false;
        
        // Morph speed controls (multiplicative)
        {
            float prevSpeed = morphSpeed;
            if (window.isKeyPressed(GLFW_KEY_Q)) {
                morphSpeed *= (1.0f + 1.5f * deltaTime);  // Multiply by 1 + 1.5*dt
            }
            if (window.isKeyPressed(GLFW_KEY_E)) {
                morphSpeed /= (1.0f + 1.5f * deltaTime);  // Divide by 1 + 1.5*dt
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

        // Move speed controls (multiplicative)
        {
            float prevSpeed = window.getCamera().getMoveSpeed();
            float moveSpeed = prevSpeed;

            if (window.isKeyPressed(GLFW_KEY_Z)) {
                moveSpeed *= (1.0f + 1.5f * deltaTime);
            }
            if (window.isKeyPressed(GLFW_KEY_C)) {
                moveSpeed /= (1.0f + 1.5f * deltaTime);
            }

            window.getCamera().setMoveSpeed(moveSpeed);
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
    
    // Check for --cpu flag
    bool useCPU = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--cpu") {
            useCPU = true;
            break;
        }
    }
    
    std::cout << "Starting Fractal Explorer..." << std::endl;
    std::cout << "Mode: " << (useCPU ? "CPU" : "CUDA GPU") << std::endl;
    
    try {
        FractalExplorer explorer(width, height, useCPU);
        explorer.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "Exiting..." << std::endl;
    return 0;
}

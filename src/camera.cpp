#include "../include/camera.h"
#include <cstring>

Camera::Camera(int width, int height) {
    params.width = width;
    params.height = height;
    params.fov = 60.0f;
    
    // Initial position
    position = make_float3(0, 0, 5);
    yaw = -90.0f;
    pitch = 0.0f;
    
    updateTransform();
}

void Camera::updateTransform() {
    // Calculate direction vectors
    float yawRad = yaw * 3.14159265f / 180.0f;
    float pitchRad = pitch * 3.14159265f / 180.0f;
    
    float3 forward;
    forward.x = cosf(yawRad) * cosf(pitchRad);
    forward.y = sinf(pitchRad);
    forward.z = sinf(yawRad) * cosf(pitchRad);
    forward = normalize(forward);
    
    float3 worldUp = make_float3(0, 1, 0);
    float3 right = normalize(cross(forward, worldUp));
    float3 up = normalize(cross(right, forward));
    
    // Build transform matrix (camera to world)
    // Column-major order for compatibility
    params.transform.m[0] = right.x;
    params.transform.m[1] = right.y;
    params.transform.m[2] = right.z;
    params.transform.m[3] = 0;
    
    params.transform.m[4] = up.x;
    params.transform.m[5] = up.y;
    params.transform.m[6] = up.z;
    params.transform.m[7] = 0;
    
    params.transform.m[8] = -forward.x;
    params.transform.m[9] = -forward.y;
    params.transform.m[10] = -forward.z;
    params.transform.m[11] = 0;
    
    params.transform.m[12] = position.x;
    params.transform.m[13] = position.y;
    params.transform.m[14] = position.z;
    params.transform.m[15] = 1;
}

void Camera::update(float deltaTime) {
    float velocity = moveSpeed * deltaTime;
    
    // Calculate direction vectors
    float yawRad = yaw * 3.14159265f / 180.0f;
    float pitchRad = pitch * 3.14159265f / 180.0f;
    
    float3 forward;
    forward.x = cosf(yawRad) * cosf(pitchRad);
    forward.y = sinf(pitchRad);
    forward.z = sinf(yawRad) * cosf(pitchRad);
    forward = normalize(forward);
    
    float3 worldUp = make_float3(0, 1, 0);
    float3 right = normalize(cross(forward, worldUp));
    
    // Move camera
    if (moveForward) position = position + forward * velocity;
    if (moveBackward) position = position - forward * velocity;
    if (moveLeft) position = position - right * velocity;
    if (moveRight) position = position + right * velocity;
    if (moveUp) position = position + worldUp * velocity;
    if (moveDown) position = position - worldUp * velocity;
    
    updateTransform();
}

void Camera::processMouseMove(double xpos, double ypos) {
    if (firstMouse) {
        lastMouseX = xpos;
        lastMouseY = ypos;
        firstMouse = false;
        return;
    }
    
    float xoffset = (xpos - lastMouseX) * lookSpeed;
    float yoffset = (lastMouseY - ypos) * lookSpeed; // Reversed since y-coordinates go from bottom to top
    
    lastMouseX = xpos;
    lastMouseY = ypos;
    
    yaw += xoffset;
    pitch += yoffset;
    
    // Constrain pitch
    if (pitch > 89.0f) pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;
    
    updateTransform();
}

void Camera::processMouseButton(int button, int action) {
    // Can be used for click interactions if needed
}

#pragma once

#include "types.h"
#include <cmath>

class Camera {
public:
    Camera(int width, int height);
    
    void update(float deltaTime);
    void processKeyboard(int key, int action);
    void processMouseMove(double xpos, double ypos);
    void processMouseButton(int button, int action);
    
    CameraParams getParams() const { return params; }
    
    // Movement
    void setMoveForward(bool moving) { moveForward = moving; }
    void setMoveBackward(bool moving) { moveBackward = moving; }
    void setMoveLeft(bool moving) { moveLeft = moving; }
    void setMoveRight(bool moving) { moveRight = moving; }
    void setMoveUp(bool moving) { moveUp = moving; }
    void setMoveDown(bool moving) { moveDown = moving; }
    
private:
    void updateTransform();
    
    CameraParams params;
    
    // Position and orientation
    float3 position;
    float yaw;   // Horizontal rotation
    float pitch; // Vertical rotation
    
    // Movement state
    bool moveForward = false;
    bool moveBackward = false;
    bool moveLeft = false;
    bool moveRight = false;
    bool moveUp = false;
    bool moveDown = false;
    
    // Movement parameters
    float moveSpeed = 2.0f;
    float lookSpeed = 0.1f;
    
    // Mouse tracking
    double lastMouseX = 0;
    double lastMouseY = 0;
    bool firstMouse = true;
};

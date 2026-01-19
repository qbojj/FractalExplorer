#pragma once

#include <GL/glew.h>
#include "camera.h"
#include <GLFW/glfw3.h>

class Window {
public:
    Window(int width, int height, const char* title);
    ~Window();
    
    bool shouldClose() const;
    void pollEvents();
    void display(const uchar3* pixels);
    double getTime() const;
    bool isKeyPressed(int key) const;
    
    Camera& getCamera() { return camera; }
    
private:
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    
    GLFWwindow* window;
    Camera camera;
    int width, height;
    GLuint textureID;
};

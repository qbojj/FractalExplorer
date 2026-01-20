#include "../include/window.h"
#include <iostream>
#include <cstring>

Window::Window(int w, int h, const char* title)
    : camera(w, h), width(w), height(h) {
    
    // Check for display
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        std::cerr << "Make sure DISPLAY is set (echo $DISPLAY)" << std::endl;
        exit(1);
    }
    
    // Create window with compatibility profile for legacy OpenGL functions
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);  // Ensure window is visible
    
    window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        exit(1);
    }
    
    glfwMakeContextCurrent(window);
    glfwSetWindowUserPointer(window, this);
    
    // GLFW 3.x handles OpenGL loading automatically - no GLEW needed
    // Print OpenGL info
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;
    
    // Set callbacks
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    
    // Capture mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    
    // Create texture for display
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    
    std::cout << "Window created: " << width << "x" << height << std::endl;
}

Window::~Window() {
    glDeleteTextures(1, &textureID);
    glfwDestroyWindow(window);
    glfwTerminate();
}

bool Window::shouldClose() const {
    return glfwWindowShouldClose(window);
}

void Window::pollEvents() {
    glfwPollEvents();
}

void Window::display(const uchar3* pixels) {
    // Upload texture
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);
    
    // Clear and render
    glClear(GL_COLOR_BUFFER_BIT);
    
    // Simple fullscreen quad rendering
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, textureID);
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1, 1, -1, 1, -1, 1);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    glBegin(GL_QUADS);
    glTexCoord2f(0, 1); glVertex2f(-1, -1);
    glTexCoord2f(1, 1); glVertex2f( 1, -1);
    glTexCoord2f(1, 0); glVertex2f( 1,  1);
    glTexCoord2f(0, 0); glVertex2f(-1,  1);
    glEnd();
    
    glfwSwapBuffers(window);
}

double Window::getTime() const {
    return glfwGetTime();
}

bool Window::isKeyPressed(int key) const {
    return glfwGetKey(window, key) == GLFW_PRESS;
}

void Window::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    Window* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
    
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    
    bool isPressed = (action == GLFW_PRESS || action == GLFW_REPEAT);
    
    if (key == GLFW_KEY_W) win->camera.setMoveForward(isPressed);
    if (key == GLFW_KEY_S) win->camera.setMoveBackward(isPressed);
    if (key == GLFW_KEY_A) win->camera.setMoveLeft(isPressed);
    if (key == GLFW_KEY_D) win->camera.setMoveRight(isPressed);
    if (key == GLFW_KEY_SPACE) win->camera.setMoveUp(isPressed);
    if (key == GLFW_KEY_LEFT_SHIFT) win->camera.setMoveDown(isPressed);
    
    if (action == GLFW_RELEASE) {
        if (key == GLFW_KEY_W) win->camera.setMoveForward(false);
        if (key == GLFW_KEY_S) win->camera.setMoveBackward(false);
        if (key == GLFW_KEY_A) win->camera.setMoveLeft(false);
        if (key == GLFW_KEY_D) win->camera.setMoveRight(false);
        if (key == GLFW_KEY_SPACE) win->camera.setMoveUp(false);
        if (key == GLFW_KEY_LEFT_SHIFT) win->camera.setMoveDown(false);
    }
}

void Window::cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    Window* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
    win->camera.processMouseMove(xpos, ypos);
}

void Window::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    Window* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
    win->camera.processMouseButton(button, action);
}

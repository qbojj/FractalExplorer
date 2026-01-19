cmake_minimum_required(VERSION 3.18)
project(FractalExplorer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90)

# Find packages
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --expt-relaxed-constexpr")

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${OPENGL_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
)

# CUDA source files
set(CUDA_SOURCES
    src/cuda/raymarcher.cu
    src/cuda/fractals.cu
)

# C++ source files
set(CPP_SOURCES
    src/camera.cpp
    src/window.cpp
)

# Main file (compiled with CUDA)
set(MAIN_SOURCE
    src/main.cpp
)

# Mark main.cpp to be compiled with CUDA compiler
set_source_files_properties(${MAIN_SOURCE} PROPERTIES LANGUAGE CUDA)

# Create executable
add_executable(fractal-explorer
    ${CPP_SOURCES}
    ${CUDA_SOURCES}
    ${MAIN_SOURCE}
)

target_link_libraries(fractal-explorer
    ${OPENGL_LIBRARIES}
    GLEW::GLEW
    glfw
)

# Set CUDA properties
set_target_properties(fractal-explorer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

message(STATUS "=================================================")
message(STATUS "  Fractal Explorer Configuration")
message(STATUS "=================================================")
message(STATUS "  CMake version: ${CMAKE_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Version: ${CUDA_VERSION}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "=================================================")

# Fractal Explorer - CUDA Ray Marcher

Real-time GPU-accelerated fractal visualization using CUDA and ray marching with signed distance functions (SDFs). Explore infinite 3D fractal landscapes with interactive camera controls.

**This is a CUDA implementation inspired by PySpace's fractal ray marching approach.**

## Features

- **Real-time CUDA Ray Marching** - 60+ FPS at 1280x720
- **Multiple Fractal Types** - Mandelbox, Menger Sponge, Sierpinski, Tree Planet
- **Smooth Fractal Morphing** - Interpolate between fractals in real-time
- **Auto-Cycle Mode** - Automatically transition through all fractals
- **Interactive Camera** - WASD movement, mouse look
- **Sphere Tracing Algorithm** - Efficient distance field rendering
- **Dynamic Lighting** - Phong shading with shadows and glow effects

## Quick Start

### Prerequisites

- NVIDIA GPU (Compute Capability 6.0+)
- CUDA Toolkit 11.0+
- CMake 3.18+
- GLFW3, GLEW, OpenGL

### Installation (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt-get install cmake nvidia-cuda-toolkit libglfw3-dev libglew-dev

# Clone and build
cd /path/to/proj
chmod +x build.sh
./build.sh

# Run
./build/fractal-explorer
```

### Custom Resolution

```bash
./build/fractal-explorer 1920 1080
```

## Controls

### Movement
- **WASD** - Move camera forward/left/backward/right
- **Space** - Move up
- **Shift** - Move down
- **Mouse** - Look around

### Fractal Controls
- **1-4** - Switch to fractal type (with smooth morphing)
  - 1: Mandelbox
  - 2: Menger Sponge
  - 3: Sierpinski Tetrahedron
  - 4: Tree Planet
- **Tab** - Toggle auto-cycle mode (automatic fractal transitions)
- **+/=** - Increase morph speed
- **-** - Decrease morph speed
- **ESC** - Exit

## Project Structure

```
src/
├── cuda/
│   ├── fractals.cu       # Distance estimators and folding operations
│   └── raymarcher.cu     # CUDA ray marching kernel
├── main.cpp              # Main application
├── camera.cpp            # Camera controller
└── window.cpp            # OpenGL window management

include/
├── types.h               # Math types and structures
├── camera.h              # Camera interface
└── window.h              # Window interface
```

## Implemented Fractals

### Mandelbox (Type 0)
Box and sphere folding iterations creating complex cubic structures.
- **Parameters**: scale=2.0, iterations=15
- **Best for**: Exploring intricate box-like formations

### Menger Sponge (Type 1)
Recursive cube subdivision with Menger folding.
- **Parameters**: scale=3.0, iterations=8
- **Best for**: Classic fractal sponge structure

### Sierpinski Tetrahedron (Type 2)
Tetrahedral symmetry folding.
- **Parameters**: scale=2.0, iterations=10
- **Best for**: Pyramid-like recursive geometry

### Tree Planet (Type 3)
Hybrid fractal combining rotation, abs, and Menger folds.
- **Parameters**: scale=1.3, iterations=30
- **Best for**: Organic tree-like structures

## Technical Details

### Ray Marching Algorithm

The core rendering uses sphere tracing:

```cuda
float t = 0;
for (int i = 0; i < MAX_STEPS; i++) {
    float3 pos = origin + direction * t;
    float dist = sceneSDF(pos);
    
    if (dist < EPSILON) return HIT;
    t += dist;  // Safe to step by distance
    if (t > MAX_DIST) break;
}
```

### Folding Operations

Fractals are generated through iterated transformations:

- **Box Fold**: `clamp(p, -r, r) * 2 - p`
- **Sphere Fold**: Inversion around min/max radius
- **Menger Fold**: Coordinate permutations for cube subdivision
- **Sierpinski Fold**: Tetrahedral symmetry reflections

### Performance

- **Resolution**: 1280x720
- **Target FPS**: 60+
- **Max Ray Steps**: 128
- **Speedup vs CPU**: ~100-200x

## Future Extensions

- [ ] Parameter interpolation system
- [ ] World mutation engine
- [ ] Additional fractal types
- [ ] Real-time parameter controls via UI
- [ ] Screenshot/video recording
- [ ] Collision detection for gameplay mechanics

## References

- PySpace (HackerPoet) - Original GLSL implementation inspiration
- Hart, J.C. (1996) - "Sphere Tracing" paper
- Quilez, I. - Distance function library and techniques

## License

MIT License - See LICENSE file for details

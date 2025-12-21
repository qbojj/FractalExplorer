# Procedural Fractal World Generator

Real-time GPU-accelerated fractal world generation and visualization using ray marching with signed distance functions (SDFs). The system supports smooth interpolation between different procedurally generated worlds, enabling real-time mutation and exploration of infinite fractal landscapes.

The worlds will be 3 dimentsional fractals defined by combinations of folding operations, allowing for complex and detailed structures that can be rendered efficiently using ray marching techniques on the GPU.

As functions of offsets and parameters are continuous, we can interpolate between different fractal worlds by smoothly varying these parameters over time. This allows for real-time morphing between distinct fractal configurations, creating a dynamic and visually engaging experience.

## Key Features

- **Ray Marching Engine:** Sphere tracing algorithm for rendering implicit surfaces
- **Interpolation System:** Smooth morphing between different worlds in real-time
- **Mutation Engine:** Procedural parameter evolution and exploration
- **CUDA Acceleration:** High-performance GPU implementation for real-time frame rates

Interpolatable procedural worlds - seamlessly morphing between different fractal configurations while maintaining visual coherence.


### possible extensions:
- implementing collision detection and physics interactions within the fractal worlds:
    - it may be possible to make a game where player is tasked to move to the exit on a fractal world "Super Monkey Ball" style

---

More details in [README-INIT.md](README-INIT.md)

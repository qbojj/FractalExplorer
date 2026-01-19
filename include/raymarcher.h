#pragma once

#include "types.h"

// Forward declare the CUDA kernel
__global__ void renderKernel(
    uchar3* output,
    const CameraParams camera,
    const FractalParams fractalParams,
    const RenderParams renderParams
);

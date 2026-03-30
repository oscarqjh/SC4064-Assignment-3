#include "pipeline.cuh"
#include <cmath>
#include <cstdio>

// ─────────────────────────────────────────────────────────────────────────────
// Gaussian kernel weights (normalised by 273).
// Stored as a constant array in GPU constant memory for fast broadcast reads.
// ─────────────────────────────────────────────────────────────────────────────
__constant__ float c_gauss[5][5] = {
    { 1.f/273,  4.f/273,  7.f/273,  4.f/273,  1.f/273 },
    { 4.f/273, 16.f/273, 26.f/273, 16.f/273,  4.f/273 },
    { 7.f/273, 26.f/273, 41.f/273, 26.f/273,  7.f/273 },
    { 4.f/273, 16.f/273, 26.f/273, 16.f/273,  4.f/273 },
    { 1.f/273,  4.f/273,  7.f/273,  4.f/273,  1.f/273 },
};


// ═════════════════════════════════════════════════════════════════════════════
// STAGE 1 — Gaussian Blur (shared memory tiling with halo cells)
// ═════════════════════════════════════════════════════════════════════════════
//
// Background:
//   Each output pixel is a weighted average of its 5x5 neighbourhood.
//   Neighbouring output pixels share input pixels, so loading the input tile
//   into shared memory reduces global memory traffic significantly.
//   The shared tile must be larger than the output tile by GAUSS_RADIUS pixels
//   on every side — these extra pixels are called "halo cells".
//
// Shared memory layout:
//
//   ┌────────────────────────────┐  ← (TILE_W + 2*GAUSS_RADIUS) wide
//   │  halo  │  halo   │  halo   │  } GAUSS_RADIUS rows of halo
//   ├────────┼─────────┼─────────┤
//   │  halo  │ OUTPUT  │  halo   │  } TILE_H rows of output pixels
//   ├────────┼─────────┼─────────┤
//   │  halo  │  halo   │  halo   │  } GAUSS_RADIUS rows of halo
//   └────────────────────────────┘
//
// Your tasks:
//   1. Declare shared memory with the correct halo-extended dimensions.
//   2. Map each thread to a global (x, y) position.
//   3. Load the centre pixels AND halo pixels into shared memory cooperatively.
//      (Some threads may need to load more than one pixel.)
//   4. __syncthreads() before any computation.
//   5. Apply the 5x5 convolution from shared memory for in-bounds threads.
//   6. Write the result to `out`.
//
// ─────────────────────────────────────────────────────────────────────────────
__global__ void gaussianBlurKernel(
    const uint8_t* __restrict__ in,
    uint8_t*       __restrict__ out,
    int width, int height)
{
    // Shared memory tile dimensions (centre + halo on each side).
    const int SMEM_W = TILE_W + 2 * GAUSS_RADIUS;
    const int SMEM_H = TILE_H + 2 * GAUSS_RADIUS;

    __shared__ float smem[TILE_H + 2 * GAUSS_RADIUS][TILE_W + 2 * GAUSS_RADIUS];

    int out_x = blockIdx.x * TILE_W + threadIdx.x;
    int out_y = blockIdx.y * TILE_H + threadIdx.y;

    // Top-left corner of the shared memory region in global coords
    int tile_start_x = blockIdx.x * TILE_W - GAUSS_RADIUS;
    int tile_start_y = blockIdx.y * TILE_H - GAUSS_RADIUS;

    // Strided cooperative load — some threads load >1 pixel since
    // SMEM_W*SMEM_H (20*20=400) > TILE_W*TILE_H (16*16=256)
    int tid = threadIdx.y * TILE_W + threadIdx.x;
    int block_size = TILE_W * TILE_H;

    for (int idx = tid; idx < SMEM_W * SMEM_H; idx += block_size) {
        int sy = idx / SMEM_W;
        int sx = idx % SMEM_W;
        int gx = min(max(tile_start_x + sx, 0), width  - 1);
        int gy = min(max(tile_start_y + sy, 0), height - 1);
        smem[sy][sx] = (float)in[gy * width + gx];
    }

    __syncthreads();

    // 5x5 convolution from shared memory
    if (out_x < width && out_y < height) {
        float sum = 0.f;
        for (int ki = 0; ki < 5; ki++) {
            for (int kj = 0; kj < 5; kj++) {
                sum += c_gauss[ki][kj] * smem[threadIdx.y + ki][threadIdx.x + kj];
            }
        }
        out[out_y * width + out_x] = (uint8_t)min(max((int)roundf(sum), 0), 255);
    }

}


// ═════════════════════════════════════════════════════════════════════════════
// STAGE 2 — Sobel Edge Detection
// ═════════════════════════════════════════════════════════════════════════════
//
// Background
//   Two 3x3 kernels (Gx, Gy) measure intensity gradient in x and y directions.
//   Gradient magnitude = sqrt(Gx^2 + Gy^2), clamped to [0, 255].
//
//   Gx = [[-1, 0, 1],     Gy = [[ 1,  2,  1],
//         [-2, 0, 2],           [ 0,  0,  0],
//         [-1, 0, 1]]           [-1, -2, -1]]
//
// Both Gx and Gy must be computed in this single kernel.
// Shared memory tiling is optional but encouraged.
// Use clamp-to-edge for boundary pixels.
//
// ─────────────────────────────────────────────────────────────────────────────
__global__ void sobelKernel(
    const uint8_t* __restrict__ in,
    uint8_t*       __restrict__ out,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Load 3x3 patch (clamp at edges)
    int p[3][3];
    for (int ki = -1; ki <= 1; ki++) {
        for (int kj = -1; kj <= 1; kj++) {
            int nx = min(max(x + kj, 0), width  - 1);
            int ny = min(max(y + ki, 0), height - 1);
            p[ki + 1][kj + 1] = (int)in[ny * width + nx];
        }
    }

    // Gx
    int sx = -p[0][0] + p[0][2]
           - 2*p[1][0] + 2*p[1][2]
           - p[2][0] + p[2][2];

    // Gy
    int sy =  p[0][0] + 2*p[0][1] + p[0][2]
            - p[2][0] - 2*p[2][1] - p[2][2];

    float mag = sqrtf((float)(sx * sx + sy * sy));
    out[y * width + x] = (uint8_t)min(max((int)mag, 0), 255);
}


// ═════════════════════════════════════════════════════════════════════════════
// STAGE 3A — Histogram Kernel
// ═════════════════════════════════════════════════════════════════════════════
//
// Background:
//   Count how many pixels have each intensity value (0–255).
//   Many threads will try to increment the same bin simultaneously,
//   so atomic operations are required.
//
// `hist` is a device array of 256 unsigned ints, zero-initialised before launch.
//
// Optimisation hint (optional, but worth attempting):
//   Use a per-block shared memory histogram (256 unsigned ints), accumulate
//   locally with __atomicAdd on shared memory, then flush to global memory
//   once per block. This reduces contention on the 256 global counters.
//
// ─────────────────────────────────────────────────────────────────────────────
__global__ void histogramKernel(
    const uint8_t*  __restrict__ in,
    unsigned int*   hist,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    uint8_t val = in[y * width + x];
    atomicAdd(&hist[val], 1u);
}

// ═════════════════════════════════════════════════════════════════════════════
// STAGE 3B — CDF on host (solution given in multigpu.cu)
// ═════════════════════════════════════════════════════════════════════════════

// ═════════════════════════════════════════════════════════════════════════════
// STAGE 3C — Equalisation Kernel
// ═════════════════════════════════════════════════════════════════════════════
//
// Background:
//   Remap each pixel using:
//     new_val = round((CDF[old_val] - cdf_min) / (W*H - cdf_min) * 255)
//
// `cdf` is a device array of 256 floats from thrust::exclusive_scan, so:
//  cdf[i] = number of pixels with intensity STRICTLY LESS THAN i, cdf[0] = 0.
//  cdf_min is the first non-zero value in cdf[], found on the host after the scan.
//
// ─────────────────────────────────────────────────────────────────────────────
__global__ void equalizeKernel(
    const uint8_t* __restrict__ in,
    uint8_t*       __restrict__ out,
    const float*   cdf,
    float          cdf_min,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    uint8_t old_val = in[y * width + x];
    float num   = cdf[old_val] - cdf_min;
    float denom = (float)(width * height) - cdf_min;
    float new_val = roundf(num / denom * 255.f);
    out[y * width + x] = (uint8_t)min(max((int)new_val, 0), 255);
}

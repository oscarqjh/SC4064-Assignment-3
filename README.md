# SC4064 Assignment 3 — GPU-Accelerated Image Processing Pipeline

CUDA image processing pipeline: Gaussian blur -> Sobel edge detection -> histogram equalisation. Processes batches of 512x512 greyscale PGM images. Uses CUDA streams and multi-GPU batch splitting.

## System Requirements

| Component  | Specification                |
|------------|------------------------------|
| OS         | Linux (tested on RHEL 9.2)   |
| GPU        | NVIDIA H100 80GB HBM3       |
| Driver     | 550.90.07                    |
| CUDA       | 13.1                         |
| Arch       | sm_90 (compute capability 9.0) |

## Environment Setup

```bash
# Create conda environment (if not already set up from Assignment 2)
conda env create -f environment.yaml

# Activate
conda activate cuda_build

# Set environment variables
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH

# Verify
nvcc --version
```

## Project Structure

```
SC4064-Assignment-3/
  assignment3/
    Makefile                  — Build system (targets H100 sm_90)
    src/
      main.cu                 — Entry point (batch loading, timing)
      pipeline.cu             — Three pipeline stage kernels (implemented)
      pipeline.cuh            — Kernel declarations and TILE_W/TILE_H constants
      multigpu.cu             — Multi-GPU batch distribution (implemented)
      multigpu.cuh            — Multi-GPU interface
      pgm_io.cu               — PGM loader/saver (provided)
      benchmark.cu            — Per-kernel timing and roofline data
      bench_clean.cu          — End-to-end pipeline timing (no I/O)
      bench_e2e.cu            — End-to-end timing with allocation overhead
    data/
      input/                  — 20 synthetic test images (512x512 PGM)
      expected_output/        — Reference outputs for all three stages
    scripts/
      check_outputs.py        — Automated correctness checker
    output/                   — Pipeline outputs (created by make run)
  report/
    paper.tex                 — Performance analysis report (4 pages)
    paper.pdf                 — Compiled report
    assets/roofline.pdf       — Roofline plot figure
    scripts/roofline_plot.py  — Script to regenerate roofline plot
```

## Building and Running

All commands should be run from the `assignment3/` directory with the conda environment activated.

```bash
cd assignment3
```

### Build

```bash
CONDA_PREFIX=/mnt/aigc/users/qianjianheng/miniconda3/envs/cuda_build make build
```

### Run (multi-GPU mode, default)

```bash
CONDA_PREFIX=/mnt/aigc/users/qianjianheng/miniconda3/envs/cuda_build make run
```

### Run (single-GPU mode)

```bash
CONDA_PREFIX=/mnt/aigc/users/qianjianheng/miniconda3/envs/cuda_build make run-single
```

### Correctness Check

```bash
# Full pipeline check (stage 3 output vs reference)
CONDA_PREFIX=/mnt/aigc/users/qianjianheng/miniconda3/envs/cuda_build make check

# Per-stage checks
python3 scripts/check_outputs.py --output output --reference data/expected_output/stage-1 --tolerance 2
python3 scripts/check_outputs.py --output output --reference data/expected_output/stage-2 --tolerance 2
python3 scripts/check_outputs.py --output output --reference data/expected_output/stage-3 --tolerance 2
```

### Benchmarking

```bash
# Per-kernel timing + roofline data
CONDA_PREFIX=/mnt/aigc/users/qianjianheng/miniconda3/envs/cuda_build \
  $CONDA_PREFIX/bin/nvcc -std=c++17 -gencode arch=compute_90,code=sm_90 -O2 \
  -Xcompiler -Wall,-Wextra -o benchmark \
  src/benchmark.cu src/pipeline.cu src/multigpu.cu src/pgm_io.cu
./benchmark --runs 200

# End-to-end pipeline timing (no I/O)
CONDA_PREFIX=/mnt/aigc/users/qianjianheng/miniconda3/envs/cuda_build \
  $CONDA_PREFIX/bin/nvcc -std=c++17 -gencode arch=compute_90,code=sm_90 -O2 \
  -Xcompiler -Wall,-Wextra -o bench_clean \
  src/bench_clean.cu src/pipeline.cu src/multigpu.cu src/pgm_io.cu
./bench_clean --runs 200
```

### Report

```bash
cd ../report
pdflatex paper.tex && pdflatex paper.tex   # run twice for references
```

## Implementation Summary

### Stage 1 — Gaussian Blur (`pipeline.cu`)

5x5 convolution with shared-memory tiling. Each block cooperatively loads a (TILE_W+4) x (TILE_H+4) region (including halo) into shared memory using a strided loop. Clamp-to-edge for boundaries. Kernel weights in `__constant__` memory.

**Result:** 20/20 images pass, MaxDiff = 0 (pixel-exact match).

### Stage 2 — Sobel Edge Detection (`pipeline.cu`)

Both Gx and Gy 3x3 kernels in one launch. Convolution done in integer, then `(int)sqrtf(Gx^2 + Gy^2)` for the magnitude (truncated, not rounded). Clamp-to-edge at boundaries.

**Result:** 20/20 images pass, MaxDiff = 0.

### Stage 3 — Histogram Equalisation (`pipeline.cu`)

Three sub-steps:
- **3A — Histogram:** `atomicAdd` to count pixel intensities into 256 bins.
- **3B — CDF:** `thrust::exclusive_scan` computes cumulative distribution. `cdf_min` found on host.
- **3C — Equalise:** Remaps pixels via `round((CDF[v] - cdf_min) / (W*H - cdf_min) * 255)`.

**Result:** 20/20 images pass, MaxDiff = 0.

### CUDA Streams (`multigpu.cu`)

One stream per image — H->D, all kernels, D->H all go on the same stream so they're serialised per image but can overlap across images. The one catch is `thrust::exclusive_scan` runs on the default stream, so we have to `cudaStreamSynchronize` before it, which kills the overlap in practice.

### Multi-GPU (`multigpu.cu`)

Uses `cudaGetDeviceCount` to check available GPUs, splits the batch in half across GPU 0 and GPU 1. Falls back to single GPU if only one is available. Each GPU has its own buffers and streams.

## Performance Results (H100 80GB HBM3)

| Kernel | Time/image (ms) | AI (FLOPs/byte) | Achieved (GFLOPs/s) |
|--------|-----------------|------------------|---------------------|
| Gaussian Blur | 0.0088 | 24.50 | 1,461 |
| Sobel | 0.0073 | 0.40 | 145 |
| Histogram | 0.0645 | — (integer) | — |
| Equalise | 0.0069 | 1.33 | 11 |

- **Bottleneck:** Histogram kernel (59% of kernel time) due to atomic contention.
- **Stream overlap:** None observed — thrust CDF serialises all streams.
- **Multi-GPU speedup:** 1.00x (sequential dispatch; would need threading for parallelism).

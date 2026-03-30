// benchmark.cu — Measure per-kernel timing and end-to-end pipeline timing
// for the performance analysis report.

#include "pipeline.cuh"
#include "multigpu.cuh"
#include "pgm_io.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// Measure individual kernel times using CUDA events (single image, averaged)
// ─────────────────────────────────────────────────────────────────────────────
static void benchmark_kernels(const char* input_path, int n_runs)
{
    uint8_t* tmp = nullptr;
    int W, H;
    if (!pgm_load(std::string(input_path), &tmp, &W, &H)) {
        fprintf(stderr, "Failed to load %s\n", input_path);
        return;
    }

    size_t img_bytes  = (size_t)W * H;
    size_t hist_bytes = 256 * sizeof(unsigned int);
    size_t cdf_bytes  = 256 * sizeof(float);

    // Allocate device buffers
    uint8_t *d_in, *d_blur, *d_edges, *d_out;
    unsigned int* d_hist;
    float* d_cdf;
    cudaMalloc(&d_in,    img_bytes);
    cudaMalloc(&d_blur,  img_bytes);
    cudaMalloc(&d_edges, img_bytes);
    cudaMalloc(&d_out,   img_bytes);
    cudaMalloc(&d_hist,  hist_bytes);
    cudaMalloc(&d_cdf,   cdf_bytes);

    cudaMemcpy(d_in, tmp, img_bytes, cudaMemcpyHostToDevice);
    delete[] tmp;

    dim3 block(TILE_W, TILE_H);
    dim3 grid((W + TILE_W - 1) / TILE_W, (H + TILE_H - 1) / TILE_H);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float ms;

    // ── Warmup ───────────────────────────────────────────────────────────
    for (int r = 0; r < 3; r++) {
        gaussianBlurKernel<<<grid, block>>>(d_in, d_blur, W, H);
        sobelKernel<<<grid, block>>>(d_blur, d_edges, W, H);
        cudaMemset(d_hist, 0, hist_bytes);
        histogramKernel<<<grid, block>>>(d_edges, d_hist, W, H);
        equalizeKernel<<<grid, block>>>(d_edges, d_out, d_cdf, 0.f, W, H);
    }
    cudaDeviceSynchronize();

    // ── Gaussian Blur ────────────────────────────────────────────────────
    float total_blur = 0.f;
    for (int r = 0; r < n_runs; r++) {
        cudaEventRecord(start);
        gaussianBlurKernel<<<grid, block>>>(d_in, d_blur, W, H);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        total_blur += ms;
    }

    // ── Sobel ────────────────────────────────────────────────────────────
    float total_sobel = 0.f;
    for (int r = 0; r < n_runs; r++) {
        cudaEventRecord(start);
        sobelKernel<<<grid, block>>>(d_blur, d_edges, W, H);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        total_sobel += ms;
    }

    // ── Histogram ────────────────────────────────────────────────────────
    float total_hist = 0.f;
    for (int r = 0; r < n_runs; r++) {
        cudaMemset(d_hist, 0, hist_bytes);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        histogramKernel<<<grid, block>>>(d_edges, d_hist, W, H);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        total_hist += ms;
    }

    // ── CDF (thrust) ────────────────────────────────────────────────────
    // Prepare a valid histogram first
    cudaMemset(d_hist, 0, hist_bytes);
    histogramKernel<<<grid, block>>>(d_edges, d_hist, W, H);
    cudaDeviceSynchronize();

    float total_cdf = 0.f;
    for (int r = 0; r < n_runs; r++) {
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        thrust::device_ptr<unsigned int> hist_ptr(d_hist);
        thrust::device_ptr<float>        cdf_ptr(d_cdf);
        thrust::exclusive_scan(hist_ptr, hist_ptr + 256, cdf_ptr);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        total_cdf += ms;
    }

    // Find cdf_min
    float h_cdf[256];
    cudaMemcpy(h_cdf, d_cdf, cdf_bytes, cudaMemcpyDeviceToHost);
    float cdf_min = 0.f;
    for (int b = 0; b < 256; b++) {
        if (h_cdf[b] > 0.f) { cdf_min = h_cdf[b]; break; }
    }

    // ── Equalise ─────────────────────────────────────────────────────────
    float total_eq = 0.f;
    for (int r = 0; r < n_runs; r++) {
        cudaEventRecord(start);
        equalizeKernel<<<grid, block>>>(d_edges, d_out, d_cdf, cdf_min, W, H);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        total_eq += ms;
    }

    printf("\n=== Per-Kernel Timing (image: %dx%d, averaged over %d runs) ===\n", W, H, n_runs);
    printf("  Gaussian Blur:     %.4f ms\n", total_blur / n_runs);
    printf("  Sobel:             %.4f ms\n", total_sobel / n_runs);
    printf("  Histogram:         %.4f ms\n", total_hist / n_runs);
    printf("  CDF (thrust):      %.4f ms\n", total_cdf / n_runs);
    printf("  Equalise:          %.4f ms\n", total_eq / n_runs);
    printf("  Total kernels:     %.4f ms\n",
           (total_blur + total_sobel + total_hist + total_cdf + total_eq) / n_runs);

    // ── Compute achieved GFLOPs/s for roofline ──────────────────────────
    int total_pixels = W * H;

    // Gaussian blur: per pixel: 25 multiply + 25 add + 1 roundf = 51 FLOPs
    // (roundf counted as 1 FLOP)
    // Actually: 25 mul + 24 add (summing 25 terms = 24 additions) + 1 roundf
    // Let's be precise: sum of c_gauss[ki][kj]*smem = 25 muls + 24 adds = 49 FLOPs
    // Plus clamp: 1 roundf. But roundf/min/max are not FP arithmetic.
    // Strict: 25 mul + 24 add = 49 FLOPs per pixel
    double blur_flops = (double)total_pixels * 49.0;
    double blur_time_s = (total_blur / n_runs) * 1e-3;
    double blur_gflops = blur_flops / blur_time_s / 1e9;

    // Sobel: per pixel: Gx has 6 int muls + 5 int adds (but these are INT, not FP)
    // FP ops: sqrtf(sx*sx + sy*sy) = 2 mul + 1 add + 1 sqrt = 4 FLOPs
    // But sx, sy are computed in integer. Only the final magnitude is float.
    // Strict FP count: 2 mul (sx*sx, sy*sy) + 1 add + 1 sqrt = 4 FLOPs per pixel
    double sobel_flops = (double)total_pixels * 4.0;
    double sobel_time_s = (total_sobel / n_runs) * 1e-3;
    double sobel_gflops = sobel_flops / sobel_time_s / 1e9;

    // Histogram: per pixel: 1 atomicAdd (integer). 0 FLOPs (no floating-point).
    // We can report throughput in pixels/s instead.
    double hist_time_s = (total_hist / n_runs) * 1e-3;

    // Equalise: per pixel: cdf[old_val]-cdf_min = 1 sub, W*H-cdf_min = 1 sub,
    // num/denom = 1 div, *255 = 1 mul, roundf = 1 (not counted). = 4 FLOPs
    double eq_flops = (double)total_pixels * 4.0;
    double eq_time_s = (total_eq / n_runs) * 1e-3;
    double eq_gflops = eq_flops / eq_time_s / 1e9;

    // Combined hist+eq FLOPs
    double histeq_flops = eq_flops; // histogram has 0 FP ops
    double histeq_time_s = hist_time_s + eq_time_s + (total_cdf / n_runs) * 1e-3;
    double histeq_gflops = histeq_flops / histeq_time_s / 1e9;

    printf("\n=== Roofline Data ===\n");

    // Bytes: Gaussian blur with shared memory tiling
    // Global memory: read entire input image (W*H bytes) + write output (W*H bytes) = 2*W*H bytes
    // Per pixel: 2 bytes global memory traffic (with perfect tiling)
    // Without tiling (naive): each pixel read 25 times = 25 + 1 = 26 bytes/pixel
    double blur_bytes_tiled = (double)total_pixels * 2.0;
    double blur_bytes_naive = (double)total_pixels * 26.0;
    double blur_ai_tiled = blur_flops / blur_bytes_tiled;
    double blur_ai_naive = blur_flops / blur_bytes_naive;

    printf("  Gaussian Blur:\n");
    printf("    FLOPs/pixel: 49 (25 mul + 24 add)\n");
    printf("    Bytes/pixel (tiled): 2 (1 read + 1 write)\n");
    printf("    Bytes/pixel (naive): 26 (25 reads + 1 write)\n");
    printf("    AI (tiled): %.2f FLOPs/byte\n", blur_ai_tiled);
    printf("    AI (naive): %.2f FLOPs/byte\n", blur_ai_naive);
    printf("    Achieved: %.2f GFLOPs/s\n", blur_gflops);
    printf("    Effective bandwidth: %.2f GB/s (tiled), %.2f GB/s (naive)\n",
           blur_bytes_tiled / blur_time_s / 1e9,
           blur_bytes_naive / blur_time_s / 1e9);

    // Sobel bytes: read input (W*H) + write output (W*H) = 2 bytes/pixel (with tiling)
    // Naive: 9 reads + 1 write = 10 bytes/pixel
    // Our implementation: no shared memory, so naive: 9 reads + 1 write = 10 bytes/pixel
    // But cache may help. Effective: ~10 bytes/pixel
    double sobel_bytes = (double)total_pixels * 10.0; // naive (no shared mem)
    double sobel_ai = sobel_flops / sobel_bytes;

    printf("  Sobel:\n");
    printf("    FP FLOPs/pixel: 4 (2 mul + 1 add + 1 sqrt)\n");
    printf("    Bytes/pixel: 10 (9 reads + 1 write, no shared mem)\n");
    printf("    AI: %.2f FLOPs/byte\n", sobel_ai);
    printf("    Achieved: %.2f GFLOPs/s\n", sobel_gflops);
    printf("    Effective bandwidth: %.2f GB/s\n",
           sobel_bytes / sobel_time_s / 1e9);

    // Histogram + Equalise combined
    // Histogram: read W*H bytes, atomicAdd to 256*4 bytes. Bytes = W*H + negligible = ~1 byte/pixel read + ~0 write
    // Equalise: read input W*H bytes + read cdf (256*4, negligible) + write W*H bytes = 2 bytes/pixel
    // Combined: ~3 bytes/pixel for hist+eq
    double histeq_bytes = (double)total_pixels * 3.0;
    double histeq_ai = histeq_flops / histeq_bytes;

    printf("  Histogram + Equalise:\n");
    printf("    FLOPs/pixel: 4 (equalize only; histogram is integer)\n");
    printf("    Bytes/pixel: ~3 (1 read hist + 2 read/write eq)\n");
    printf("    AI: %.2f FLOPs/byte\n", histeq_ai);
    printf("    Achieved: %.2f GFLOPs/s\n", histeq_gflops);
    printf("    Effective bandwidth: %.2f GB/s\n",
           histeq_bytes / histeq_time_s / 1e9);

    // H100 specs
    printf("\n=== H100 80GB HBM3 Theoretical Peaks ===\n");
    printf("  Peak FP32: ~989 TFLOPS (with tensor cores) / ~67 TFLOPS (CUDA cores)\n");
    printf("  Peak memory bandwidth: ~3.35 TB/s\n");
    printf("  Ridge point (CUDA cores): ~20 FLOPs/byte\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in); cudaFree(d_blur); cudaFree(d_edges);
    cudaFree(d_out); cudaFree(d_hist); cudaFree(d_cdf);
}

// ─────────────────────────────────────────────────────────────────────────────
// Measure end-to-end pipeline time (excluding I/O) for single vs multi GPU
// ─────────────────────────────────────────────────────────────────────────────
static void benchmark_pipeline(const char* input_dir, int n_runs)
{
    // Load all images
    std::vector<std::string> input_files;
    for (auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.path().extension() == ".pgm")
            input_files.push_back(entry.path().string());
    }
    std::sort(input_files.begin(), input_files.end());

    std::vector<ImageEntry> batch;
    for (auto& path : input_files) {
        ImageEntry e;
        e.input_path = path;
        fs::path p(path);
        e.output_path = "/dev/null"; // discard output
        uint8_t* tmp = nullptr;
        if (!pgm_load(path, &tmp, &e.width, &e.height)) continue;
        size_t n = (size_t)e.width * e.height;
        cudaMallocHost(&e.host_in, n);
        cudaMallocHost(&e.host_out, n);
        memcpy(e.host_in, tmp, n);
        memset(e.host_out, 0, n);
        delete[] tmp;
        batch.push_back(e);
    }

    printf("\n=== End-to-End Pipeline Timing (%zu images, %d runs) ===\n",
           batch.size(), n_runs);

    // Single GPU warmup + timing
    {
        // Warmup
        std::vector<ImageEntry> warmup_batch = batch;
        run_pipeline_singlegpu(warmup_batch);

        double total_ms = 0.0;
        for (int r = 0; r < n_runs; r++) {
            std::vector<ImageEntry> b = batch;
            cudaDeviceSynchronize();
            auto t0 = std::chrono::high_resolution_clock::now();
            run_pipeline_singlegpu(b);
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            total_ms += ms;
        }
        printf("  Single-GPU: %.4f ms\n", total_ms / n_runs);
    }

    // Multi GPU warmup + timing
    {
        std::vector<ImageEntry> warmup_batch = batch;
        run_pipeline_multigpu(warmup_batch);

        double total_ms = 0.0;
        for (int r = 0; r < n_runs; r++) {
            std::vector<ImageEntry> b = batch;
            cudaDeviceSynchronize();
            auto t0 = std::chrono::high_resolution_clock::now();
            run_pipeline_multigpu(b);
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            total_ms += ms;
        }
        printf("  Multi-GPU:  %.4f ms\n", total_ms / n_runs);
    }

    for (auto& e : batch) {
        cudaFreeHost(e.host_in);
        cudaFreeHost(e.host_out);
    }
}

int main(int argc, char** argv)
{
    int n_runs = 100;
    const char* input_dir = "data/input";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--runs") == 0 && i+1 < argc) n_runs = atoi(argv[++i]);
        if (strcmp(argv[i], "--input") == 0 && i+1 < argc) input_dir = argv[++i];
    }

    // Find first image for per-kernel benchmark
    std::string first_image;
    for (auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.path().extension() == ".pgm") {
            first_image = entry.path().string();
            break;
        }
    }

    if (!first_image.empty()) {
        benchmark_kernels(first_image.c_str(), n_runs);
    }

    benchmark_pipeline(input_dir, n_runs);

    return 0;
}

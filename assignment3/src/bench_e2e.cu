// bench_e2e.cu — Measure end-to-end GPU-only time (no pgm_save) for
// single-GPU vs multi-GPU comparison.

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

// Stripped-down version of process_batch_on_device that skips pgm_save
static void process_batch_no_io(std::vector<ImageEntry>& sub_batch, int device_id)
{
    cudaSetDevice(device_id);

    int W = sub_batch[0].width;
    int H = sub_batch[0].height;
    size_t img_bytes = (size_t)W * H * sizeof(uint8_t);
    size_t hist_bytes = 256 * sizeof(unsigned int);
    size_t cdf_bytes  = 256 * sizeof(float);
    int    n_images  = (int)sub_batch.size();

    std::vector<uint8_t*>      d_in(n_images),  d_blur(n_images);
    std::vector<uint8_t*>      d_edges(n_images), d_out(n_images);
    std::vector<unsigned int*> d_hist(n_images);
    std::vector<float*>        d_cdf(n_images);

    for (int i = 0; i < n_images; i++) {
        cudaMalloc(&d_in[i],    img_bytes);
        cudaMalloc(&d_blur[i],  img_bytes);
        cudaMalloc(&d_edges[i], img_bytes);
        cudaMalloc(&d_out[i],   img_bytes);
        cudaMalloc(&d_hist[i],  hist_bytes);
        cudaMalloc(&d_cdf[i],   cdf_bytes);
    }

    std::vector<cudaStream_t> streams(n_images);
    for (int i = 0; i < n_images; i++) cudaStreamCreate(&streams[i]);

    dim3 block(TILE_W, TILE_H);
    dim3 grid((W + TILE_W - 1) / TILE_W, (H + TILE_H - 1) / TILE_H);

    for (int i = 0; i < n_images; i++) {
        cudaMemcpyAsync(d_in[i], sub_batch[i].host_in, img_bytes,
                        cudaMemcpyHostToDevice, streams[i]);
        gaussianBlurKernel<<<grid, block, 0, streams[i]>>>(d_in[i], d_blur[i], W, H);
        sobelKernel<<<grid, block, 0, streams[i]>>>(d_blur[i], d_edges[i], W, H);
        cudaMemsetAsync(d_hist[i], 0, hist_bytes, streams[i]);
        histogramKernel<<<grid, block, 0, streams[i]>>>(d_edges[i], d_hist[i], W, H);

        cudaStreamSynchronize(streams[i]);
        thrust::device_ptr<unsigned int> hist_ptr(d_hist[i]);
        thrust::device_ptr<float>        cdf_ptr(d_cdf[i]);
        thrust::exclusive_scan(hist_ptr, hist_ptr + 256, cdf_ptr);

        float h_cdf[256];
        cudaMemcpy(h_cdf, d_cdf[i], cdf_bytes, cudaMemcpyDeviceToHost);
        float cdf_min = 0.f;
        for (int b = 0; b < 256; b++) {
            if (h_cdf[b] > 0.f) { cdf_min = h_cdf[b]; break; }
        }

        equalizeKernel<<<grid, block, 0, streams[i]>>>(
            d_edges[i], d_out[i], d_cdf[i], cdf_min, W, H);
        cudaMemcpyAsync(sub_batch[i].host_out, d_out[i], img_bytes,
                        cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < n_images; i++) cudaStreamSynchronize(streams[i]);

    for (int i = 0; i < n_images; i++) {
        cudaStreamDestroy(streams[i]);
        cudaFree(d_in[i]); cudaFree(d_blur[i]); cudaFree(d_edges[i]);
        cudaFree(d_out[i]); cudaFree(d_hist[i]); cudaFree(d_cdf[i]);
    }
}

int main(int argc, char** argv)
{
    int n_runs = 50;
    const char* input_dir = "data/input";
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--runs") == 0 && i+1 < argc) n_runs = atoi(argv[++i]);
        if (strcmp(argv[i], "--input") == 0 && i+1 < argc) input_dir = argv[++i];
    }

    std::vector<std::string> input_files;
    for (auto& entry : fs::directory_iterator(input_dir))
        if (entry.path().extension() == ".pgm")
            input_files.push_back(entry.path().string());
    std::sort(input_files.begin(), input_files.end());

    std::vector<ImageEntry> batch;
    for (auto& path : input_files) {
        ImageEntry e;
        e.input_path = path;
        e.output_path = ""; // not used
        uint8_t* tmp = nullptr;
        if (!pgm_load(path, &tmp, &e.width, &e.height)) continue;
        size_t n = (size_t)e.width * e.height;
        cudaMallocHost(&e.host_in, n);
        cudaMallocHost(&e.host_out, n);
        memcpy(e.host_in, tmp, n);
        delete[] tmp;
        batch.push_back(e);
    }

    printf("Loaded %zu images (%dx%d), %d runs each\n",
           batch.size(), batch[0].width, batch[0].height, n_runs);

    // Warmup
    {
        std::vector<ImageEntry> b = batch;
        process_batch_no_io(b, 0);
    }

    // Single GPU
    {
        double total = 0;
        for (int r = 0; r < n_runs; r++) {
            std::vector<ImageEntry> b = batch;
            cudaDeviceSynchronize();
            auto t0 = std::chrono::high_resolution_clock::now();
            process_batch_no_io(b, 0);
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            total += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        printf("Single-GPU (no I/O): %.4f ms\n", total / n_runs);
    }

    // Multi GPU (sequential, matching current implementation)
    {
        int num_gpus = 0;
        cudaGetDeviceCount(&num_gpus);
        if (num_gpus >= 2) {
            // Warmup GPU 1
            int half = (int)batch.size() / 2;
            std::vector<ImageEntry> sub1(batch.begin() + half, batch.end());
            process_batch_no_io(sub1, 1);

            double total = 0;
            for (int r = 0; r < n_runs; r++) {
                int h = (int)batch.size() / 2;
                std::vector<ImageEntry> s0(batch.begin(), batch.begin() + h);
                std::vector<ImageEntry> s1(batch.begin() + h, batch.end());
                cudaDeviceSynchronize();
                auto t0 = std::chrono::high_resolution_clock::now();
                process_batch_no_io(s0, 0);
                process_batch_no_io(s1, 1);
                cudaSetDevice(0); cudaDeviceSynchronize();
                cudaSetDevice(1); cudaDeviceSynchronize();
                auto t1 = std::chrono::high_resolution_clock::now();
                total += std::chrono::duration<double, std::milli>(t1 - t0).count();
            }
            printf("Multi-GPU  (no I/O, sequential): %.4f ms (%d GPUs)\n", total / n_runs, num_gpus);
        }
    }

    for (auto& e : batch) { cudaFreeHost(e.host_in); cudaFreeHost(e.host_out); }
    return 0;
}

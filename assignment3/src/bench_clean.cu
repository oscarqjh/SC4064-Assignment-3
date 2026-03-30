// bench_clean.cu — Pre-allocate everything, measure only kernel + transfer time
#include "pipeline.cuh"
#include "multigpu.cuh"
#include "pgm_io.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

namespace fs = std::filesystem;

int main(int argc, char** argv)
{
    int n_runs = 200;
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
        e.output_path = "";
        uint8_t* tmp = nullptr;
        if (!pgm_load(path, &tmp, &e.width, &e.height)) continue;
        size_t n = (size_t)e.width * e.height;
        cudaMallocHost(&e.host_in, n);
        cudaMallocHost(&e.host_out, n);
        memcpy(e.host_in, tmp, n);
        delete[] tmp;
        batch.push_back(e);
    }

    int W = batch[0].width, H = batch[0].height;
    size_t img_bytes = (size_t)W * H;
    size_t hist_bytes = 256 * sizeof(unsigned int);
    size_t cdf_bytes  = 256 * sizeof(float);
    int N = (int)batch.size();

    printf("=== Clean benchmark: %d images (%dx%d), %d runs ===\n", N, W, H, n_runs);

    // ----- SINGLE GPU: pre-allocate -----
    cudaSetDevice(0);
    std::vector<uint8_t*> d_in(N), d_blur(N), d_edges(N), d_out(N);
    std::vector<unsigned int*> d_hist(N);
    std::vector<float*> d_cdf(N);
    std::vector<cudaStream_t> streams(N);

    for (int i = 0; i < N; i++) {
        cudaMalloc(&d_in[i], img_bytes);
        cudaMalloc(&d_blur[i], img_bytes);
        cudaMalloc(&d_edges[i], img_bytes);
        cudaMalloc(&d_out[i], img_bytes);
        cudaMalloc(&d_hist[i], hist_bytes);
        cudaMalloc(&d_cdf[i], cdf_bytes);
        cudaStreamCreate(&streams[i]);
    }

    dim3 block(TILE_W, TILE_H);
    dim3 grid((W + TILE_W - 1) / TILE_W, (H + TILE_H - 1) / TILE_H);

    // Warmup
    for (int i = 0; i < N; i++) {
        cudaMemcpyAsync(d_in[i], batch[i].host_in, img_bytes, cudaMemcpyHostToDevice, streams[i]);
        gaussianBlurKernel<<<grid, block, 0, streams[i]>>>(d_in[i], d_blur[i], W, H);
    }
    cudaDeviceSynchronize();

    // Measure single-GPU pipeline (transfers + kernels + thrust CDF)
    cudaEvent_t t_start, t_stop;
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_stop);

    float total_single = 0.f;
    for (int r = 0; r < n_runs; r++) {
        cudaEventRecord(t_start);
        for (int i = 0; i < N; i++) {
            cudaMemcpyAsync(d_in[i], batch[i].host_in, img_bytes,
                            cudaMemcpyHostToDevice, streams[i]);
            gaussianBlurKernel<<<grid, block, 0, streams[i]>>>(d_in[i], d_blur[i], W, H);
            sobelKernel<<<grid, block, 0, streams[i]>>>(d_blur[i], d_edges[i], W, H);
            cudaMemsetAsync(d_hist[i], 0, hist_bytes, streams[i]);
            histogramKernel<<<grid, block, 0, streams[i]>>>(d_edges[i], d_hist[i], W, H);

            cudaStreamSynchronize(streams[i]);
            thrust::device_ptr<unsigned int> hp(d_hist[i]);
            thrust::device_ptr<float> cp(d_cdf[i]);
            thrust::exclusive_scan(hp, hp + 256, cp);

            float h_cdf[256];
            cudaMemcpy(h_cdf, d_cdf[i], cdf_bytes, cudaMemcpyDeviceToHost);
            float cdf_min = 0.f;
            for (int b = 0; b < 256; b++)
                if (h_cdf[b] > 0.f) { cdf_min = h_cdf[b]; break; }

            equalizeKernel<<<grid, block, 0, streams[i]>>>(
                d_edges[i], d_out[i], d_cdf[i], cdf_min, W, H);
            cudaMemcpyAsync(batch[i].host_out, d_out[i], img_bytes,
                            cudaMemcpyDeviceToHost, streams[i]);
        }
        for (int i = 0; i < N; i++) cudaStreamSynchronize(streams[i]);
        cudaEventRecord(t_stop);
        cudaEventSynchronize(t_stop);
        float ms;
        cudaEventElapsedTime(&ms, t_start, t_stop);
        total_single += ms;
    }
    printf("Single-GPU (pre-alloc, no I/O): %.4f ms\n", total_single / n_runs);

    // Clean up GPU 0
    for (int i = 0; i < N; i++) {
        cudaStreamDestroy(streams[i]);
        cudaFree(d_in[i]); cudaFree(d_blur[i]); cudaFree(d_edges[i]);
        cudaFree(d_out[i]); cudaFree(d_hist[i]); cudaFree(d_cdf[i]);
    }

    // ----- MULTI GPU: pre-allocate on both GPUs -----
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus < 2) {
        printf("Only %d GPU — skipping multi-GPU benchmark\n", num_gpus);
    } else {
        int half = N / 2;
        int sizes[2] = {half, N - half};

        // Pre-allocate on each GPU
        struct GPUData {
            std::vector<uint8_t*> d_in, d_blur, d_edges, d_out;
            std::vector<unsigned int*> d_hist;
            std::vector<float*> d_cdf;
            std::vector<cudaStream_t> streams;
        } gpu[2];

        for (int g = 0; g < 2; g++) {
            cudaSetDevice(g);
            int n = sizes[g];
            gpu[g].d_in.resize(n); gpu[g].d_blur.resize(n);
            gpu[g].d_edges.resize(n); gpu[g].d_out.resize(n);
            gpu[g].d_hist.resize(n); gpu[g].d_cdf.resize(n);
            gpu[g].streams.resize(n);
            for (int i = 0; i < n; i++) {
                cudaMalloc(&gpu[g].d_in[i], img_bytes);
                cudaMalloc(&gpu[g].d_blur[i], img_bytes);
                cudaMalloc(&gpu[g].d_edges[i], img_bytes);
                cudaMalloc(&gpu[g].d_out[i], img_bytes);
                cudaMalloc(&gpu[g].d_hist[i], hist_bytes);
                cudaMalloc(&gpu[g].d_cdf[i], cdf_bytes);
                cudaStreamCreate(&gpu[g].streams[i]);
            }
        }

        // Warmup both
        for (int g = 0; g < 2; g++) {
            cudaSetDevice(g);
            for (int i = 0; i < sizes[g]; i++) {
                int bi = (g == 0) ? i : half + i;
                cudaMemcpyAsync(gpu[g].d_in[i], batch[bi].host_in, img_bytes,
                                cudaMemcpyHostToDevice, gpu[g].streams[i]);
                gaussianBlurKernel<<<grid, block, 0, gpu[g].streams[i]>>>(
                    gpu[g].d_in[i], gpu[g].d_blur[i], W, H);
            }
            cudaDeviceSynchronize();
        }

        // Sequential multi-GPU (current implementation)
        cudaSetDevice(0);
        cudaEvent_t ms_start, ms_stop;
        cudaEventCreate(&ms_start);
        cudaEventCreate(&ms_stop);

        auto run_on_gpu = [&](int g) {
            cudaSetDevice(g);
            int n = sizes[g];
            for (int i = 0; i < n; i++) {
                int bi = (g == 0) ? i : half + i;
                cudaMemcpyAsync(gpu[g].d_in[i], batch[bi].host_in, img_bytes,
                                cudaMemcpyHostToDevice, gpu[g].streams[i]);
                gaussianBlurKernel<<<grid, block, 0, gpu[g].streams[i]>>>(
                    gpu[g].d_in[i], gpu[g].d_blur[i], W, H);
                sobelKernel<<<grid, block, 0, gpu[g].streams[i]>>>(
                    gpu[g].d_blur[i], gpu[g].d_edges[i], W, H);
                cudaMemsetAsync(gpu[g].d_hist[i], 0, hist_bytes, gpu[g].streams[i]);
                histogramKernel<<<grid, block, 0, gpu[g].streams[i]>>>(
                    gpu[g].d_edges[i], gpu[g].d_hist[i], W, H);

                cudaStreamSynchronize(gpu[g].streams[i]);
                thrust::device_ptr<unsigned int> hp(gpu[g].d_hist[i]);
                thrust::device_ptr<float> cp(gpu[g].d_cdf[i]);
                thrust::exclusive_scan(hp, hp + 256, cp);

                float h_cdf[256];
                cudaMemcpy(h_cdf, gpu[g].d_cdf[i], cdf_bytes, cudaMemcpyDeviceToHost);
                float cdf_min = 0.f;
                for (int b = 0; b < 256; b++)
                    if (h_cdf[b] > 0.f) { cdf_min = h_cdf[b]; break; }

                equalizeKernel<<<grid, block, 0, gpu[g].streams[i]>>>(
                    gpu[g].d_edges[i], gpu[g].d_out[i], gpu[g].d_cdf[i], cdf_min, W, H);
                cudaMemcpyAsync(batch[bi].host_out, gpu[g].d_out[i], img_bytes,
                                cudaMemcpyDeviceToHost, gpu[g].streams[i]);
            }
            for (int i = 0; i < n; i++) cudaStreamSynchronize(gpu[g].streams[i]);
        };

        float total_multi_seq = 0.f;
        for (int r = 0; r < n_runs; r++) {
            cudaSetDevice(0);
            cudaEventRecord(ms_start);
            run_on_gpu(0);
            run_on_gpu(1);
            cudaSetDevice(0);
            cudaEventRecord(ms_stop);
            cudaEventSynchronize(ms_stop);
            float ms;
            cudaEventElapsedTime(&ms, ms_start, ms_stop);
            total_multi_seq += ms;
        }
        printf("Multi-GPU  (sequential, pre-alloc): %.4f ms\n", total_multi_seq / n_runs);

        // Cleanup
        for (int g = 0; g < 2; g++) {
            cudaSetDevice(g);
            for (int i = 0; i < sizes[g]; i++) {
                cudaStreamDestroy(gpu[g].streams[i]);
                cudaFree(gpu[g].d_in[i]); cudaFree(gpu[g].d_blur[i]);
                cudaFree(gpu[g].d_edges[i]); cudaFree(gpu[g].d_out[i]);
                cudaFree(gpu[g].d_hist[i]); cudaFree(gpu[g].d_cdf[i]);
            }
        }
        cudaEventDestroy(ms_start);
        cudaEventDestroy(ms_stop);
    }

    cudaEventDestroy(t_start);
    cudaEventDestroy(t_stop);
    for (auto& e : batch) { cudaFreeHost(e.host_in); cudaFreeHost(e.host_out); }
    return 0;
}

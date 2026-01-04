#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "mhc_layer.cuh"
#include "mhc_types.h"
#include "utils.cuh"

using namespace mhc;

void run_static_benchmark(int B, int C, int n, int bench_runs, L2Flusher& flusher, bool use_tc) {
    float* h_x_expanded = (float*)malloc(B * n * C * sizeof(float));
    floatX* h_rmsnorm_weight = (floatX*)malloc(C * sizeof(floatX));
    float* h_H_pre = (float*)malloc(n * sizeof(float));
    float* h_H_post = (float*)malloc(n * sizeof(float));
    float* h_H_res = (float*)malloc(n * n * sizeof(float));

    srand(42);
    for (int i = 0; i < B * n * C; i++) {
        h_x_expanded[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
    for (int i = 0; i < C; i++) {
        h_rmsnorm_weight[i] = (floatX)((float)rand() / RAND_MAX * 0.5f + 0.75f);
    }
    for (int i = 0; i < n; i++) {
        h_H_pre[i] = 0.0f;
        h_H_post[i] = 0.0f;
    }
    for (int i = 0; i < n * n; i++) {
        h_H_res[i] = 0.01f * ((float)rand() / RAND_MAX * 2.0f - 1.0f);
    }

    float* d_x_expanded;
    CHECK_CUDA(cudaMalloc(&d_x_expanded, B * n * C * sizeof(float)));
    CHECK_CUDA(
        cudaMemcpy(d_x_expanded, h_x_expanded, B * n * C * sizeof(float), cudaMemcpyHostToDevice));

    MHCLayerConfig cfg;
    cfg.batch_size = B;
    cfg.hidden_dim = C;
    cfg.expansion_rate = n;
    cfg.sinkhorn_iters = 20;
    cfg.eps = 1e-5f;
    cfg.use_pdl = true;
    cfg.use_dynamic_h = false;

    MHCLayer layer;
    layer.init(cfg);
    layer.use_tc_mix = use_tc;
    layer.set_weights(h_rmsnorm_weight, h_H_pre, h_H_post, h_H_res);
    layer.sync();

    layer.forward_device(d_x_expanded);
    layer.sync();

    size_t bytes_io = (size_t)B * n * C * sizeof(float) * 3;

    BenchTimer timer;
    float total_time = 0.0f;

    for (int i = 0; i < bench_runs; i++) {
        flusher.flush();

        timer.record_start();
        layer.forward_device(d_x_expanded);
        timer.record_stop();
        total_time += timer.elapsed_ms();
    }

    float avg_time_ms = total_time / bench_runs;
    float throughput = B / (avg_time_ms / 1000.0f);
    float bw = (bytes_io / 1e9f) / (avg_time_ms / 1e3f);

    printf("%6d %6d %4d %8s %6s %12.2f %14.0f %14.0f\n", B, C, n, "static",
           use_tc ? "TC" : "CUDA CORE", avg_time_ms * 1000.0f, throughput, bw);

    layer.destroy();
    cudaFree(d_x_expanded);
    free(h_x_expanded);
    free(h_rmsnorm_weight);
    free(h_H_pre);
    free(h_H_post);
    free(h_H_res);
}

void run_dynamic_benchmark(int B, int C, int n, int bench_runs, L2Flusher& flusher) {
    float* h_x_expanded = (float*)malloc(B * n * C * sizeof(float));
    floatX* h_rmsnorm_weight = (floatX*)malloc(C * sizeof(floatX));

    int nC = n * C;
    int total_H_dim = n + n + n * n;
    floatX* h_phi = (floatX*)malloc(total_H_dim * nC * sizeof(floatX));
    float* h_b_pre = (float*)malloc(n * sizeof(float));
    float* h_b_post = (float*)malloc(n * sizeof(float));
    float* h_b_res = (float*)malloc(n * n * sizeof(float));

    srand(42);
    for (int i = 0; i < B * n * C; i++) {
        h_x_expanded[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
    for (int i = 0; i < C; i++) {
        h_rmsnorm_weight[i] = (floatX)((float)rand() / RAND_MAX * 0.5f + 0.75f);
    }
    for (int i = 0; i < total_H_dim * nC; i++) {
        h_phi[i] = (floatX)((float)rand() / RAND_MAX * 0.1f - 0.05f);
    }
    for (int i = 0; i < n; i++) {
        h_b_pre[i] = 0.0f;
        h_b_post[i] = 0.0f;
    }
    for (int i = 0; i < n * n; i++) {
        h_b_res[i] = 0.01f * ((float)rand() / RAND_MAX * 2.0f - 1.0f);
    }

    float* d_x_expanded;
    CHECK_CUDA(cudaMalloc(&d_x_expanded, B * n * C * sizeof(float)));
    CHECK_CUDA(
        cudaMemcpy(d_x_expanded, h_x_expanded, B * n * C * sizeof(float), cudaMemcpyHostToDevice));

    MHCLayerConfig cfg;
    cfg.batch_size = B;
    cfg.hidden_dim = C;
    cfg.expansion_rate = n;
    cfg.sinkhorn_iters = 20;
    cfg.eps = 1e-5f;
    cfg.use_pdl = true;
    cfg.use_dynamic_h = true;

    MHCLayer layer;
    layer.init(cfg);

    floatX* h_phi_pre = h_phi;
    floatX* h_phi_post = h_phi + n * nC;
    floatX* h_phi_res = h_phi + 2 * n * nC;

    layer.set_weights_dynamic(h_rmsnorm_weight, h_phi_pre, h_phi_post, h_phi_res, h_b_pre, h_b_post,
                              h_b_res, 0.01f, 0.01f, 0.01f);
    layer.sync();

    layer.forward_device(d_x_expanded);
    layer.sync();

    size_t bytes_io = (size_t)B * n * C * sizeof(float) * 3;

    BenchTimer timer;
    float total_time = 0.0f;

    for (int i = 0; i < bench_runs; i++) {
        flusher.flush();

        timer.record_start();
        layer.forward_device(d_x_expanded);
        timer.record_stop();
        total_time += timer.elapsed_ms();
    }

    float avg_time_ms = total_time / bench_runs;
    float throughput = B / (avg_time_ms / 1000.0f);
    float bw = (bytes_io / 1e9f) / (avg_time_ms / 1e3f);

    printf("%6d %6d %4d %8s %6s %12.2f %14.0f %14.0f\n", B, C, n, "dynamic", "CUDA",
           avg_time_ms * 1000.0f, throughput, bw);

    layer.destroy();
    cudaFree(d_x_expanded);
    free(h_x_expanded);
    free(h_rmsnorm_weight);
    free(h_phi);
    free(h_b_pre);
    free(h_b_post);
    free(h_b_res);
}

int main() {
    const int bench_runs = 50;

    L2Flusher flusher;

    struct Config {
        int B;
        int C;
        int n;
    };

    Config configs[] = {
        {64, 1280, 4},  {128, 1280, 4}, {256, 1280, 4},  {320, 1280, 4},
        {64, 1920, 4},  {128, 1920, 4}, {64, 2560, 4},   {128, 2560, 4},
        {32, 1280, 32}, {64, 1280, 32}, {128, 1280, 32},
    };
    int num_configs = sizeof(configs) / sizeof(configs[0]);

    printf("MHC Layer End-to-End Benchmark\n");
    printf("==========================================================\n");
    printf("Pipeline: Aggregate(H_pre) -> RMSNorm -> Distribute(H_post) + Mix(M)\n");
    printf("Static H: H coefficients shared across batch (Sinkhorn on H_res)\n");
    printf("Dynamic H: H coefficients computed per-sample via RMSNorm + MatMul + activations\n");
    printf("Input shape: [B, n, C]\n");
    printf("PDL path: %s\n\n",
#ifdef MHC_ENABLE_PDL
           "Enabled"
#else
           "Disabled"
#endif
    );

    printf("%6s %6s %4s %8s %6s %12s %14s %14s\n", "Batch", "Hidden", "n", "Mode", "Implementation",
           "Time (us)", "Samples/sec", "Bandwidth (GB/s)");
    printf(
        "--------------------------------------------------------------------------------------\n");

    for (int c = 0; c < num_configs; c++) {
        int B = configs[c].B;
        int C = configs[c].C;
        int n = configs[c].n;

        run_static_benchmark(B, C, n, bench_runs, flusher, false);

        if (n >= 32) {
            run_static_benchmark(B, C, n, bench_runs, flusher, true);
        }

        run_dynamic_benchmark(B, C, n, bench_runs, flusher);

        printf("\n");
    }

    return 0;
}

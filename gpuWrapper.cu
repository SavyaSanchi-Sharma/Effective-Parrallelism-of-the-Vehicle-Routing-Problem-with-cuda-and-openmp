#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include "vrp.hpp"
using namespace std;

// Number of CUDA streams for overlapping
const int NUM_STREAMS = 4;

static int* d_perms = nullptr;
static double* d_dist = nullptr;
static double* d_demands = nullptr;
static double* d_cost = nullptr;

// Pinned host memory for faster transfers
static double* h_cost_pinned = nullptr;

// CUDA streams for overlapping
static cudaStream_t streams[NUM_STREAMS];

static int g_n = 0;
static int g_batchSize = 0;
static size_t g_distSize = 0;
void initGPU(const VRP& vrp, int maxBatchSize)
{
    g_n = vrp.getSize();
    g_batchSize = maxBatchSize;
    g_distSize = vrp.dist.size();

    // Allocate device memory
    cudaMalloc(&d_perms, sizeof(int) * g_batchSize * g_n);
    cudaMalloc(&d_dist, sizeof(double) * g_distSize);
    cudaMalloc(&d_demands, sizeof(double) * g_n);
    cudaMalloc(&d_cost, sizeof(double) * g_batchSize);

    // Allocate pinned host memory for faster transfers
    cudaMallocHost(&h_cost_pinned, sizeof(double) * g_batchSize);

    // Create CUDA streams for overlapping
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    cudaMemcpy(d_dist,
               vrp.dist.data(),
               sizeof(double) * g_distSize,
               cudaMemcpyHostToDevice);

    std::vector<double> demands(g_n);
    for (int i = 0; i < g_n; i++)
        demands[i] = vrp.node[i].demand;

    cudaMemcpy(d_demands,
               demands.data(),
               sizeof(double) * g_n,
               cudaMemcpyHostToDevice);
}
// Forward declare kernel
__global__ void evaluateBatchKernel(
    const int* perms,
    const double* dist,
    const double* demands,
    double* outCost,
    int n,
    int batchSize,
    double capacity);

// Forward declare VRP
class VRP;
vector<double> gpuEvaluateBatch(
    const VRP& vrp,
    const vector<vector<int>>& batchPerms)
{
    int batchSize = batchPerms.size();
    int n = vrp.getSize();

    // Flatten permutations
    vector<int> flatPerms(batchSize * n);
    for (int b = 0; b < batchSize; b++)
        for (int i = 0; i < n; i++)
            flatPerms[b*n + i] = batchPerms[b][i];

    // Calculate sub-batch size for stream overlapping
    int subBatchSize = (batchSize + NUM_STREAMS - 1) / NUM_STREAMS;
    int blockSize = 256;

    // Process sub-batches using streams for overlapping
    for (int s = 0; s < NUM_STREAMS; s++) {
        int startIdx = s * subBatchSize;
        int endIdx = min(startIdx + subBatchSize, batchSize);
        int currentSubBatch = endIdx - startIdx;

        if (currentSubBatch <= 0) break;

        // Async copy permutations for this sub-batch
        cudaMemcpyAsync(d_perms + startIdx * n,
                       flatPerms.data() + startIdx * n,
                       sizeof(int) * currentSubBatch * n,
                       cudaMemcpyHostToDevice,
                       streams[s]);

        // Launch kernel on this stream
        int gridSize = (currentSubBatch + blockSize - 1) / blockSize;
        evaluateBatchKernel<<<gridSize, blockSize, 0, streams[s]>>>(
            d_perms + startIdx * n,
            d_dist,
            d_demands,
            d_cost + startIdx,
            n,
            currentSubBatch,
            vrp.getCapacity());

        // Async copy results back using pinned memory
        cudaMemcpyAsync(h_cost_pinned + startIdx,
                       d_cost + startIdx,
                       sizeof(double) * currentSubBatch,
                       cudaMemcpyDeviceToHost,
                       streams[s]);
    }

    // Wait for all streams to complete
    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaStreamSynchronize(streams[s]);
    }

    // Copy results from pinned memory to output vector
    vector<double> costs(batchSize);
    for (int i = 0; i < batchSize; i++) {
        costs[i] = h_cost_pinned[i];
    }

    return costs;
}
void cleanupGPU()
{
    // Free device memory
    cudaFree(d_perms);
    cudaFree(d_dist);
    cudaFree(d_demands);
    cudaFree(d_cost);

    // Free pinned host memory
    if (h_cost_pinned) {
        cudaFreeHost(h_cost_pinned);
        h_cost_pinned = nullptr;
    }

    // Destroy streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
}
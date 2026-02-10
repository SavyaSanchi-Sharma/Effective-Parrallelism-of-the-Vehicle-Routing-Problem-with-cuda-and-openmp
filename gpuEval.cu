#include <cuda_runtime.h>
#include <cstddef>

__device__ double get_dist_device(const double* dist, int size, int i, int j)
{
    if (i == j) return 0.0;

    if (i > j) {
        int tmp = i;
        i = j;
        j = tmp;
    }

    size_t myoffset   = ((2 * i * size) - (i * i) + i) / 2;
    size_t correction = 2 * i + 1;

    return dist[myoffset + j - correction];
}

__global__ void evaluateBatchKernel(
    const int* perms,
    const double* dist,
    const double* demands,
    double* outCost,
    int n,
    int batchSize,
    double capacity)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batchSize) return;

    const int* myPerm = &perms[tid * n];

    double cost = 0.0;
    double residue = capacity;
    int prev = 0;

    for (int i = 0; i < n; i++)
    {
        int node = myPerm[i];
        if (node == 0) continue;

        double d = demands[node];

        if (residue >= d)
        {
            cost += get_dist_device(dist, n, prev, node);
            residue -= d;
            prev = node;
        }
        else
        {
            cost += get_dist_device(dist, n, prev, 0);
            residue = capacity;
            prev = 0;
            i--;   // retry same node
        }
    }

    cost += get_dist_device(dist, n, prev, 0);
    outCost[tid] = cost;
}
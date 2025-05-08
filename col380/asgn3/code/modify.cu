#include "modify.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

#define THREADS_PER_BLOCK 1024

__global__ void calcFreq(int *matrix, int *freq, int total, int range) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int val = matrix[idx];
        atomicAdd(&freq[val - 1], 1);
    }
}

__global__ void prefixSum(int *freq, int *prefixSum, int range) {
    extern __shared__ int shared_freq[];
    int tid = threadIdx.x;

    if (tid < range) shared_freq[tid] = freq[tid];
    __syncthreads();

    for (int offset = 1; offset < range; offset *= 2) {
        int temp = (tid >= offset) ? shared_freq[tid - offset] : 0;
        __syncthreads();
        shared_freq[tid] += temp;
        __syncthreads();
    }

    if (tid < range) prefixSum[tid] = shared_freq[tid];
}

__global__ void modifyMatrix(int *matrix, int *output, int *prefixSum, int total, int range) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int val = matrix[idx];
        int pos = atomicSub(&prefixSum[val - 1], 1);
        output[pos - 1] = val;
    }
}

vector<vector<vector<int>>> modify(vector<vector<vector<int>>>& matrices, vector<int>& ranges) {
    const int N = matrices.size();

    vector<vector<vector<int>>> result(N, vector<vector<int>>());

    vector<cudaStream_t> streams(N);
    for (size_t i = 0; i < N; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    for (size_t m = 0; m < N; ++m) {
        int rows = matrices[m].size();
        int cols = matrices[m][0].size();
        int range = ranges[m];
        int total = rows * cols;
        
        const int blocksPerGrid = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        int *h_matrix = nullptr;
        cudaHostAlloc(&h_matrix, total * sizeof(int), cudaHostAllocDefault);
        for (int i = 0; i < rows; ++i) {
            memcpy(h_matrix + i * cols, matrices[m][i].data(), cols * sizeof(int));
        }

        int *d_matrix, *d_freq, *d_prefixSum, *d_output;
        cudaMalloc(&d_matrix, total * sizeof(int));
        cudaMalloc(&d_freq, range * sizeof(int));
        cudaMalloc(&d_prefixSum, range * sizeof(int));
        cudaMalloc(&d_output, total * sizeof(int));

        cudaMemcpyAsync(d_matrix, h_matrix, total * sizeof(int), cudaMemcpyHostToDevice, streams[m]);
        cudaMemsetAsync(d_freq, 0, range * sizeof(int), streams[m]);
        cudaMemsetAsync(d_prefixSum, 0, range * sizeof(int), streams[m]);
        cudaStreamSynchronize(streams[m]);

        /* ---- Running kernels ---- */

        calcFreq<<<blocksPerGrid, THREADS_PER_BLOCK, 0, streams[m]>>>(d_matrix, d_freq, total, range);
        prefixSum<<<1, range, range * sizeof(int), streams[m]>>>(d_freq, d_prefixSum, range);
        modifyMatrix<<<blocksPerGrid, THREADS_PER_BLOCK, 0, streams[m]>>>(d_matrix, d_output, d_prefixSum, total, range);

        /* ------------------------- */

        cudaStreamSynchronize(streams[m]);
        cudaMemcpyAsync(h_matrix, d_output, total * sizeof(int), cudaMemcpyDeviceToHost, streams[m]);
        cudaStreamSynchronize(streams[m]);

        vector<vector<int>> res(rows, vector<int>(cols));
        for (int i = 0; i < rows; ++i) {
            memcpy(res[i].data(), h_matrix + i * cols, cols * sizeof(int));
        }

        result[m] = res;

        cudaFree(d_matrix);
        cudaFree(d_freq);
        cudaFree(d_prefixSum);
        cudaFree(d_output);
        cudaFreeHost(h_matrix);
    }

    for (size_t i = 0; i < N; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    return result;
}

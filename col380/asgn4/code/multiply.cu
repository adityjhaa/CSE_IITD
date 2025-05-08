#include "matrix.hpp"
#include <cuda_runtime.h>
#include <omp.h>

__global__ void
multiplyKernel(const uint *d_A, const uint *d_B, uint *d_C, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < k && col < k) {
        uint sum = 0;
        for (int i = 0; i < k; ++i) {
            sum += d_A[row * k + i] * d_B[i * k + col] % mod;
            sum %= mod;
        }
        d_C[row * k + col] = sum;
    }
}

void multiply(const Matrix &A, const Matrix &B, Matrix &C) {
    if (A.width != B.height) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    int k = A.k;
    C.height = A.height; C.width = B.width;
    C.k = k; C.blocks.clear();

    std::vector<std::pair<pii, block>> blocksA(A.blocks.begin(), A.blocks.end());
    std::map<int, std::vector<std::pair<int, const block*>>> B_col_map;
    for (const auto &blockB : B.blocks) {
        int col = blockB.first.second;
        int row = blockB.first.first;
        B_col_map[row].push_back({col, &blockB.second});
    }

    int max_threads = omp_get_num_procs();
    omp_set_num_threads(max_threads);

    std::vector<uint*> d_A_pool(max_threads, nullptr);
    std::vector<uint*> d_B_pool(max_threads, nullptr);
    std::vector<uint*> d_C_pool(max_threads, nullptr);

    size_t size = k * k * sizeof(uint);

    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        
        cudaMalloc(&d_A_pool[thread_id], size);
        cudaMalloc(&d_B_pool[thread_id], size);
        cudaMalloc(&d_C_pool[thread_id], size);

        std::map<pii, block> local_result;

        #pragma omp for schedule(dynamic, 1) nowait
        for (size_t i = 0; i < blocksA.size(); ++i) {
            const auto &blockA_pair = blocksA[i];
            const block &blockA = blockA_pair.second;
            int blockA_row = blockA_pair.first.first;
            int blockA_col = blockA_pair.first.second;

            auto it_B_cols = B_col_map.find(blockA_col);
            if (it_B_cols == B_col_map.end()) continue;

            for (const auto &col_block_pair : it_B_cols->second) {
                int blockB_col = col_block_pair.first;
                const block *blockB_ptr = col_block_pair.second;

                pii result_coord = {blockA_row, blockB_col};

                auto it_result = local_result.find(result_coord);
                if (it_result == local_result.end()) {
                    local_result[result_coord] = block(k * k, 0);
                    it_result = local_result.find(result_coord);
                }

                block &result_block = it_result->second;

                uint *d_A = d_A_pool[thread_id];
                uint *d_B = d_B_pool[thread_id];
                uint *d_C = d_C_pool[thread_id];

                cudaMemcpyAsync(d_A, blockA.data(), size, cudaMemcpyHostToDevice);
                cudaMemcpyAsync(d_B, blockB_ptr->data(), size, cudaMemcpyHostToDevice);

                dim3 threadsPerBlock(16, 16);
                dim3 numBlocks((k + threadsPerBlock.x - 1) / threadsPerBlock.x,
                              (k + threadsPerBlock.y - 1) / threadsPerBlock.y);

                multiplyKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, k);
                cudaDeviceSynchronize();
                
                block temp_result(k * k);
                cudaMemcpy(temp_result.data(), d_C, size, cudaMemcpyDeviceToHost);
                
                for (int i = 0; i < k * k; ++i) {
                    result_block[i] = (result_block[i] + temp_result[i]) % mod;
                }
            }
        }
        
        cudaFree(d_A_pool[thread_id]);
        cudaFree(d_B_pool[thread_id]);
        cudaFree(d_C_pool[thread_id]);
        
        #pragma omp critical(update_result)
        {
            for (const auto &result_pair : local_result) {
                const pii &coord = result_pair.first;
                const block &result_block = result_pair.second;
                
                auto it_C = C.blocks.find(coord);
                if (it_C == C.blocks.end()) {
                    C.blocks[coord] = result_block;
                } else {
                    block &C_block = it_C->second;
                    for (int i = 0; i < k * k; ++i) {
                        C_block[i] = (C_block[i] + result_block[i]) % mod;
                    }
                }
            }
        }
    }
    
    removeZeros(C);
}

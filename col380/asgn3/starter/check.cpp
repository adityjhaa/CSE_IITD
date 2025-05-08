#include <algorithm>
#include <cassert>
#include <omp.h>
#include <random>
#include "modify.cuh"
#include <iostream>

// It does not checks whether each modified matrix has the same multiset as the corresponding original matrix
bool check (vector<vector<vector<int>>>& upd_matrices, vector<vector<vector<int>>>& org_matrices) {
    if(org_matrices.size() ^ upd_matrices.size()) return false;
    for (int i = 0; i < org_matrices.size(); i++) {
        const auto& matrix = upd_matrices[i];
        int rows = matrix.size(), cols = matrix[0].size();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (i > 0 && matrix[i][j] < matrix[i - 1][j]) return false;
                if (j > 0 && matrix[i][j] < matrix[i][j - 1]) return false;
            }
        }
    }
    return true;
}

vector<vector<int>> gen_matrix(int range, int rows, int cols) {
    assert (range <= (int)1e9);
    vector<vector<int>> matrix(rows, vector<int> (cols));
    omp_set_num_threads(16);
#pragma omp parallel
    {
      std::mt19937 gen(omp_get_thread_num());
      std::uniform_int_distribution<int> dist(0, range);
#pragma omp for
      for (int i = 0; i < rows; i++) {
        long long foo = dist(gen);
        for (int j = 0; j < cols; j++) {
          matrix[i][j] = 1 + ((foo ^ j) % range);
        }
      }
    }
    return matrix;
}
  
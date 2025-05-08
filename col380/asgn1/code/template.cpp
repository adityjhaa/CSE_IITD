#include <iostream>
#include <set>
#include <vector>
#include <map>
#include <utility>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <random>
#include <omp.h>
#include "check.h"

using namespace std;

#define pos_block map<pair<int, int>, vector<vector<int>>>
#define vector2D vector<vector<int>>
#define vectorf vector<float>
#define vectori vector<int>
#define pii pair<int, int>

map<pair<int, int>, vector<vector<int>>> generate_matrix(int n, int m, int b) {
    int numBlocks = n/m;

    set<pii> unique_positions;

    random_device rd_g;
    mt19937 gen(rd_g());
    uniform_int_distribution<int>spawn(0, numBlocks - 1);

    while (unique_positions.size() < b) {
        int i = spawn(gen);
        int j = spawn(gen);
        unique_positions.insert({i, j});
    }

    vector<pii> positions(unique_positions.begin(), unique_positions.end());

    pos_block matrix_map;

    #pragma omp parallel
    {
    #pragma omp single
    {
    for (auto pos: positions) {
        #pragma omp task shared(matrix_map) firstprivate(pos) if(black_box())
        {
        vector2D block(m, vector<int>(m));

        random_device rd;
        mt19937 engine(rd());
        uniform_int_distribution<int> distr(0, 256);

        bool allZero = true;

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                block[i][j] = distr(engine);
                if (block[i][j] != 0) 
                    allZero = false;
            }
        }

        if (allZero) {
            int posi = distr(engine) % m;
            int posj = distr(engine) % m;

            int val = distr(engine);
            if (val == 0) val = 1;

            block[posi][posj] = val;
        }

        #pragma omp critical
        matrix_map[pos] = block;
        }
    } } }

    #pragma omp taskwait

    return matrix_map;
}

void pre_process(pos_block &matrixBlocks) {
    #pragma omp parallel
    {
    #pragma omp single
    {
    for (auto it = matrixBlocks.begin(); it != matrixBlocks.end(); ) {
        #pragma omp task firstprivate(it) if(black_box())
        {
        vector2D& block = it->second;
        bool isBlockNonZero = false;

        for (auto& row : block) {
            for (auto& value : row) {
                if (value % 5 == 0)
                    value = 0;
                if (value != 0)
                    isBlockNonZero = true;
            }
        }

        if (!isBlockNonZero)
            #pragma omp critical
            matrixBlocks.erase(it);

        }
        ++it;
    } } }
}

void remove_zeroBlocks(pos_block &matrixBlocks) {
    #pragma omp parallel   
    {
    #pragma omp single
    {
    for (auto it = matrixBlocks.begin(); it != matrixBlocks.end(); ) {
        #pragma omp task firstprivate(it) if(black_box())
        {
        vector2D& block = it->second;
        bool isBlockNonZero = false;

        for (auto& row : block) {
            for (auto& value : row) {
                if (value != 0)
                    isBlockNonZero = true;
            }
        }

        if (!isBlockNonZero)
            #pragma omp critical
            matrixBlocks.erase(it);

        }
        ++it;
    } } }
}

pos_block identity(int n, int m) {
    int numBlocks = n / m;

    pos_block result;
    for (int i = 0; i < numBlocks; ++i) {
        pii pos = {i, i};
        result[pos] = vector2D(m, vectori(m, 0));
        for (int j = 0; j < m; ++j) {
            result[pos][j][j] = 1;
        }
    }

    return result;
}

pos_block multiply(pos_block &matrix1, pos_block &matrix2, int n, int m, vectori &P, int power) {
    pos_block product_matrix;
 
    #pragma omp parallel if (black_box())
    {
    #pragma omp single
    {        
    for (auto& [pos_1, block_1]: matrix1) {
        for (auto& [pos_2, block_2]: matrix2) {
            if (pos_1.second != pos_2.first) { continue; }
            #pragma omp task firstprivate(pos_1, block_1, pos_2, block_2) if(black_box())
            {
            pii pos = {pos_1.first, pos_2.second};
            
            vector2D product(m, vectori(m, 0));

            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < m; ++j) {
                    long sum = 0;
                    for (int k = 0; k < m; ++k) {
                        int val = block_1[i][k] * block_2[k][j];
                        sum += val;
                        if (power == 2 && val != 0) {
                            int row = pos.first * m + i;
                            #pragma omp atomic
                            P[row]++;
                        }
                    }
                    product[i][j] = sum;
                }
            }

            #pragma omp critical
            if (product_matrix.find(pos) != product_matrix.end()) {
                for (int i = 0; i < m; ++i) {
                    for (int j = 0; j < m; ++j) {
                        product_matrix[pos][i][j] += product[i][j];
                    }
                }
            } else {
                product_matrix[pos] = product;
            }
            
        } }
    } } }

    #pragma omp taskwait
    return product_matrix;
}

vectorf compute_statistics(vectori &P, pos_block &blocks, int n, int m) {
    vector<float> row_statistic(n, 0.0f);
    vectori B(n, 0);

    for (auto &[pos, block]: blocks) {
        for (int i = 0; i < m; ++i) {
                int row = pos.first * m + i;
                B[row] += m;
        }
    }

    for (int i = 0; i < n; ++i) {
        if (B[i] == 0) { continue; }
        row_statistic[i] =  (float)P[i] / (float)B[i];
    }

    return row_statistic;
}

vector<float> matmul(map<pair<int, int>, vector<vector<int>>>& blocks, int n, int m, int k) {
    pre_process(blocks);
   
    if (k == 0) {
        blocks = identity(n, m);
        return vector<float>();
    } else if (k == 1) { return vector<float>(); }
    
    int numBlocks = n / m;
    vectori P(n, 0);

    if (k == 2) {
        blocks = multiply(blocks, blocks, n, m, P, 2);
        remove_zeroBlocks(blocks);
        return compute_statistics(P, blocks, n, m);
    }

    pos_block result = identity(n, m);
    pos_block base = blocks;

    while (k > 0) {
        if (k % 2 == 1) {
            result = multiply(result, base, n, m, P, 0);
        }
        base = multiply(base, base, n, m, P, 0);
        k /= 2;
    }
    blocks = result;
    
    remove_zeroBlocks(blocks);
    return vector<float>();
}

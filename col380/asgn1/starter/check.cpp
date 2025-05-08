#include <iostream>
#include <vector>
#include <map>
#include <utility>
#include <cstdlib>
#include <ctime>
#include <random>
#include <algorithm>
#include "check.h"

using namespace std;

bool black_box() {
    return true;
}

vector<vector<int>> multiply_blocks(vector<vector<int>>& block1,
                                    vector<vector<int>>& block2, int m) {
    vector<vector<int>> product(m, vector<int>(m, 0));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < m; ++k) {
                product[i][j] += block1[i][k] * block2[k][j];
            }
        }
    }
    return product;
}

void removeMultiplesOf5(map<pair<int, int>, vector<vector<int>>>& matrixBlocks) {
    for (auto it = matrixBlocks.begin(); it != matrixBlocks.end(); ) {
        vector<vector<int>>& block = it->second;
        bool isBlockNonZero = false;

        for (auto& row : block) {
            for (auto& value : row) {
                if (value % 5 == 0) {
                    value = 0;
                }
                if (value != 0) {
                    isBlockNonZero = true;
                }
            }
        }

        if (!isBlockNonZero) {
            it = matrixBlocks.erase(it);
        } else {
            ++it;
        }
    }
}

bool is_square(map<pair<int, int>, vector<vector<int>>>& matrix1,
               map<pair<int, int>, vector<vector<int>>>& matrix2, int m) {
    removeMultiplesOf5(matrix1);
    map<pair<int, int>, vector<vector<int>>> squared_result;

    for (auto& [block_pos1, block1] : matrix1) {
        for (auto& [block_pos2, block2] : matrix1) {
            if (block_pos1.second == block_pos2.first) {
                vector<vector<int>> product = multiply_blocks(block1, block2, m);
                if (squared_result.find({block_pos1.first, block_pos2.second}) != squared_result.end()) {
                    vector<vector<int>>& existing_block = squared_result[{block_pos1.first, block_pos2.second}];
                    for (int i = 0; i < m; ++i) {
                        for (int j = 0; j < m; ++j) {
                            existing_block[i][j] += product[i][j];
                        }
                    }
                } else {
                    squared_result[{block_pos1.first, block_pos2.second}] = product;
                }
            }
        }
    }
    

    if (squared_result.size() != matrix2.size()) {
        return false;
    }

    for (auto& [block_pos, block] : squared_result) {
        if (matrix2.find(block_pos) == matrix2.end()) {
            return false;
        }

        vector<vector<int>>& block2 = matrix2.at(block_pos);
        if (block != block2) {
            return false;
        }
    }

    return true;
}

bool has_non_zero_element(vector<vector<int>>& block) {
    for (auto& row : block)
        for (int val : row)
            if (val != 0)
                return true;
    return false;
}

int count_non_zero_blocks(map<pair<int, int>, vector<vector<int>>>& blocks) {
    int non_zero_count = 0;

    for (auto& entry : blocks) {
        vector<vector<int>>& block = entry.second;

        if (has_non_zero_element(block))
            non_zero_count++;
    }

    return non_zero_count;
}

void print_matrix_map(map<pair<int, int>, vector<vector<int>>>& matrix_map) {
    for (auto& entry : matrix_map) {
        cout << "Block (" << entry.first.first << ", " << entry.first.second << "):\n";
        for (auto& row : entry.second) {
            for (int val : row) {
                cout << val << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
}

int main() {
    int n = 100;
    int m = 3;
    int b = 3;
    int k = 2;

    srand(time(0));

    map<pair<int, int>, vector<vector<int>>> blocks = generate_matrix(n, m, b);
    if(count_non_zero_blocks(blocks)==blocks.size() && blocks.size()>=b)
        cout<<"You have generated the matrix correctly\n";
    else
        cout<<"You have NOT generated the matrix correctly\n";

    map<pair<int, int>, vector<vector<int>>> original_blocks = blocks;
    
    vector<float> s = matmul(blocks, n, m, k);

    bool res = is_square(original_blocks, blocks, m);
    if(res)
        cout<<"Your function computed the square correctly\n";
    else
        cout<<"Your function did NOT compute the square correctly\n";
    cout << "Size of S = " << s.size()<<endl;

    return 0;
}
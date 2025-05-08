#ifndef CHECK_H
#define CHECK_H

#include <map>
#include <vector>
#include <utility>

bool black_box();
void print_matrix_map(const std::map<std::pair<int, int>, std::vector<std::vector<int>>>& matrix_map);
std::map<std::pair<int, int>, std::vector<std::vector<int>>> generate_matrix(int n, int m, int b);
std::vector<float> matmul(std::map<std::pair<int, int>, std::vector<std::vector<int>>>& blocks, int n, int m, int k);

#endif // CHECK_H
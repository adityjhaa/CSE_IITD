#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <climits>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <stdint.h>
#include <unordered_map>
#include <vector>

#define uint uint64_t
#define mod LLONG_MAX

#define pii std::pair<int, int>
#define block std::vector<uint>

struct Matrix {
    int height, width, k;

    std::map<pii, block> blocks;

    Matrix() : height(0), width(0), k(0) {}
};

Matrix readMatrix(const std::string &filename, int k);

void writeMatrix(const std::string &filename, const Matrix &matrix);

void removeZeros(Matrix &matrix);

void multiply(const Matrix &A, const Matrix &B, Matrix &C);

#endif

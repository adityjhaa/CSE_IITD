#include "io.hpp"

void readMatrix(const std::string &filePath, double* matrix, int rows, int cols) {
    FILE* fp = fopen(filePath.c_str(), "rb");
    fread(matrix, sizeof(double), rows * cols, fp);
    fclose(fp);
}

void writeMatrix(const std::string &filePath, const double* matrix, int rows, int cols) {
    FILE* fp = fopen(filePath.c_str(), "wb");
    fwrite(matrix, sizeof(double), rows * cols, fp);
    fclose(fp);
}


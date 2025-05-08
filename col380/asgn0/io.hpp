#pragma once

#include <string>

void readMatrix(const std::string &filePath, double* matrix, int rows, int cols);
void writeMatrix(const std::string& filePath, const double* matrix, int rows, int cols);


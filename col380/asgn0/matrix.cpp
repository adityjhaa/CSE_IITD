#include "matrix.hpp"

void matrixMultiplyIJK(const double *A, const double *B, double *C, int rowsA, int colsA, int colsB) {
    int i, j, k;

    for (i = 0; i < rowsA; ++i) {
        for (j = 0; j < colsB; ++j) {
            for (k = 0; k < colsA; ++k) {
                C[i * colsB + j] += A[i * colsA + k] * B[k * colsB + j];
            }
        }
    }
}

void matrixMultiplyIKJ(const double *A, const double *B, double *C, int rowsA, int colsA, int colsB) {
    int i, j, k;

    for (i = 0; i < rowsA; ++i) {
        for (k = 0; k < colsA; ++k) {
            for (j = 0; j < colsB; ++j) {
                C[i * colsB + j] += A[i * colsA + k] * B[k * colsB + j];
            }
        }
    }
}

void matrixMultiplyJIK(const double *A, const double *B, double *C, int rowsA, int colsA, int colsB) {
    int i, j, k;

    for (j = 0; j < colsB; ++j) {
        for (i = 0; i < rowsA; ++i) {
            for (k = 0; k < colsA; ++k) {
                C[i * colsB + j] += A[i * colsA + k] * B[k * colsB + j];
            }
        }
    }
}

void matrixMultiplyJKI(const double *A, const double *B, double *C, int rowsA, int colsA, int colsB) {
    int i, j, k;

    for (j = 0; j < colsB; ++j) {
        for (k = 0; k < colsA; ++k) {
            for (i = 0; i < rowsA; ++i) {
                C[i * colsB + j] += A[i * colsA + k] * B[k * colsB + j];
            }
        }
    }
}

void matrixMultiplyKIJ(const double *A, const double *B, double *C, int rowsA, int colsA, int colsB) {
    int i, j, k;

    for (k = 0; k < colsA; ++k) {
        for (i = 0; i < rowsA; ++i) {
            for (j = 0; j < colsB; ++j) {
                C[i * colsB + j] += A[i * colsA + k] * B[k * colsB + j];
            }
        }
    }
}

void matrixMultiplyKJI(const double *A, const double *B, double *C, int rowsA, int colsA, int colsB) {
    int i, j, k;

    for (k = 0; k < colsA; ++k) {
        for (j = 0; j < colsB; ++j) {
            for (i = 0; i < rowsA; ++i) {
                C[i * colsB + j] += A[i * colsA + k] * B[k * colsB + j];
            }
        }
    }
}

#pragma once

void matrixMultiplyIJK(const double *A, const double *B, double *C, int rowsA, int colsA, int colsB);
void matrixMultiplyIKJ(const double *A, const double *B, double *C, int rowsA, int colsA, int colsB);
void matrixMultiplyJIK(const double *A, const double *B, double *C, int rowsA, int colsA, int colsB);
void matrixMultiplyJKI(const double *A, const double *B, double *C, int rowsA, int colsA, int colsB);
void matrixMultiplyKIJ(const double *A, const double *B, double *C, int rowsA, int colsA, int colsB);
void matrixMultiplyKJI(const double *A, const double *B, double *C, int rowsA, int colsA, int colsB);


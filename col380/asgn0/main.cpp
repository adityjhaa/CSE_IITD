#include <iostream>

#include "io.hpp"
#include "matrix.hpp"

int main(int argc, char *argv[]) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] << " <type> <mtx_A_rows> <mtx_A_cols> <mtx_B_cols> <input_path> <output_path>\n";
        return 1;
    }

    int type = std::stoi(argv[1]);
    int rowsA = std::stoi(argv[2]);
    int colsA = std::stoi(argv[3]);
    int colsB = std::stoi(argv[4]);
    const std::string inputPath = argv[5];
    const std::string outputPath = argv[6];


    double* A = new double[rowsA * colsA];
    double* B = new double[colsA * colsB];
    double* C = new double[rowsA * colsB]();

    readMatrix(inputPath + "mtx_A.bin", A, rowsA, colsA);
    readMatrix(inputPath + "mtx_B.bin", B, colsA, colsB);

    switch (type) {
        case 0:
            matrixMultiplyIJK(A, B, C, rowsA, colsA, colsB);
            break;
            
        case 1:
            matrixMultiplyIKJ(A, B, C, rowsA, colsA, colsB);
            break;
            
        case 2:
            matrixMultiplyJIK(A, B, C, rowsA, colsA, colsB);
            break;
            
        case 3:
            matrixMultiplyJKI(A, B, C, rowsA, colsA, colsB);
            break;
            
        case 4:
            matrixMultiplyKIJ(A, B, C, rowsA, colsA, colsB);
            break;
            
        case 5:
            matrixMultiplyKJI(A, B, C, rowsA, colsA, colsB);
            break;

        default:
            std::cerr << "Invalid type. Use 0-5 for loop orderings.\n";
            delete []A;
            delete []B;
            delete []C;
            return 1;
    }

    writeMatrix(outputPath + "mtx_C.bin", C, rowsA, colsB);

    delete[] A;
    delete[] B;
    delete[] C;
    
    return 0;
}

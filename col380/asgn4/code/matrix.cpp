#include "matrix.hpp"

Matrix readMatrix(const std::string &filename, int k) {
    Matrix matrix;
    matrix.k = k;

    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }

    file >> matrix.height >> matrix.width;

    int numBlocks;
    file >> numBlocks;

    for (int i = 0; i < numBlocks; ++i) {
        int x, y;
        file >> x >> y;

        block b;
        b.resize(k * k);
        for (int j = 0; j < k; ++j)
            for (int l = 0; l < k; ++l)
                file >> b[j * k + l];

        matrix.blocks[{x, y}] = b;
    }

    file.close();
    return matrix;
}

void writeMatrix(const std::string &filename, const Matrix &matrix) {
    int k = matrix.k;
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }

    file << matrix.height << " " << matrix.width << "\n";
    file << matrix.blocks.size() << "\n";

    for (const auto& _block : matrix.blocks) {
        file << _block.first.first << " " << _block.first.second << "\n";
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k - 1; ++j) {
                if (_block.first.first + i >= matrix.height || _block.first.second + j >= matrix.width) {
                    file << 0 << " ";
                }
                file << _block.second[i * k + j] << " ";
            }
            if (_block.first.first + i >= matrix.height || _block.first.second + k - 1 >= matrix.width)
                file << 0 << "\n";
            else
               file << _block.second[i * k + k - 1] << "\n";
        }
    }

    file.close();
}

void removeZeros(Matrix &matrix) {
    int k = matrix.k;
    for (auto it = matrix.blocks.begin(); it != matrix.blocks.end();) {
        bool allZero = true;
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                if (it->second[i * k + j] != 0) {
                    allZero = false;
                    break;
                }
            }
            if (!allZero) break;
        }

        if (allZero)
            it = matrix.blocks.erase(it);
        else
            ++it;
    }
}

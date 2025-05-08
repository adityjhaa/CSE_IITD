#include "matrix.hpp"
#include <chrono>
#include <mpi.h>

using namespace std;

Matrix multiply(const vector<Matrix>& matrices);

void sendMatrix(int dest, const Matrix &matrix);
Matrix recvMatrix(int src);

int main(int argc, char **argv) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <foldername>" << endl;
        return 1;
    }

    auto start = chrono::high_resolution_clock::now();
    
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    string foldername = argv[1];
    int N, k;
    
    if (rank == 0) {
        string sizePath = foldername + "/size";
        ifstream sizeFile(sizePath);
        
        if (!sizeFile) {
            cerr << "Error: Could not open file " << sizePath << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        sizeFile >> N >> k;
        sizeFile.close();
    }
    
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    int l_size = N / size;
    int l_rem = N % size;
    int l_start, l_end;
    if (rank < l_rem) {
        l_start = rank * (l_size + 1);
        l_end = l_start + l_size + 1;
    } else {
        l_start = rank * l_size + l_rem;
        l_end = l_start + l_size;
    }
    
    vector<Matrix> l_matrices;
    l_matrices.reserve(l_end - l_start);
    for (int fileNo = l_start; fileNo < l_end; ++fileNo) {
        Matrix mat = readMatrix(foldername + "/matrix" + to_string(fileNo + 1), k);
        l_matrices.push_back(mat);
    }
    
    Matrix localProduct = multiply(l_matrices);

    cout << "Rank " << rank << " finished local multiplication.\n";
    
    int active_ranks = size;
    int step = 0;

    while (active_ranks > 1) {
        if ((rank % (1 << (step + 1))) == (1 << step)) {
            int partner = rank - (1 << step);
            sendMatrix(partner, localProduct);
            break;
        } else if ((rank % (1 << (step + 1))) == 0) {
            int partner = rank + (1 << step);
            if (partner < size) {
                Matrix matrix2 = recvMatrix(partner);
                cout << "Rank " << rank << " received matrix from rank " << partner << "\n";
                Matrix temp;
                multiply(localProduct, matrix2, temp);
                localProduct = temp;
                cout << "Rank " << rank << " finished multiplication with rank " << partner << "\n";
            }
        }
        active_ranks = (active_ranks + 1) / 2;
        step++;
    }

    if (rank == 0) {
        writeMatrix("matrix", localProduct);
        cout << "Rank 0 wrote final result to 'matrix'" << "\n";
    }
    
    MPI_Finalize();

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    if (rank == 0)
        cout << fixed << setprecision(2) << "Time taken: " << duration << " ms\n";

    return 0;
}

Matrix multiply(const vector<Matrix>& matrices) {
    if (matrices.empty()) {
        return Matrix();
    }
    Matrix result = matrices[0];
    Matrix temp;
    for (size_t i = 1; i < matrices.size(); ++i) {
        temp.blocks.clear();

        multiply(result, matrices[i], temp);
        result = temp;
    }
    return result;
}

void sendMatrix(int dest, const Matrix &matrix) {
    int k = matrix.k;
    int numBlocks = matrix.blocks.size();
    MPI_Send(&matrix.height, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
    MPI_Send(&matrix.width, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
    MPI_Send(&matrix.k, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
    MPI_Send(&numBlocks, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
    
    for (const auto& _block : matrix.blocks) {
        const int coord[2] = {_block.first.first, _block.first.second};
        MPI_Send(coord, 2, MPI_INT, dest, 0, MPI_COMM_WORLD);
        
        
        block b = _block.second;
        MPI_Send(b.data(), k * k, MPI_UINT64_T, dest, 0, MPI_COMM_WORLD);
    }
}

Matrix recvMatrix(int src) {
    Matrix matrix;
    int numBlocks, k;
    MPI_Recv(&matrix.height, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&matrix.width, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&matrix.k, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&numBlocks, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    k = matrix.k;
    
    for (int i = 0; i < numBlocks; ++i) {
        int coord[2];
        MPI_Recv(coord, 2, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


        block b;
        b.resize(k * k);
        MPI_Recv(b.data(), k * k, MPI_UINT64_T, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        matrix.blocks[{coord[0], coord[1]}] = b;
    }

    return matrix;
}

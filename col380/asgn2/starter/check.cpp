#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <algorithm>
#include <ctime>  
#include "template.hpp"


using namespace std; 

void readGraphFile(string filename, vector<pair<int,int>>& edges, map<int,int>& colors) {
    ifstream file(filename);
    int V, E;
    if (!(file >> V >> E)) {
        cerr << "Error reading file: " << filename << endl;
        return;
    }
    
    // Read vertex colors
    for(int i = 0; i < V; i++) {
        int vertex, color;
        if (!(file >> vertex >> color)) {
            cerr << "Error reading vertex colors in: " << filename << endl;
            return;
        }
        colors[vertex] = color;
    }
    
    // Read edges
    for(int i = 0; i < E; i++) {
        int from, to;
        if (!(file >> from >> to)) {
            cerr << "Error reading edges in: " << filename << endl;
            return;
        }
        edges.push_back({from, to});
    }
    
    file.close();
}

pair<vector<pair<int,int>>, map<int,int>> read_graph_part(int rank, int size, string foldername) {
    vector<pair<int,int>> edges;
    map<int,int> colors;
    for(int i = rank; i < 40; i += size) {
        string filename = foldername + "/graph_" + to_string(i) + ".txt";
        readGraphFile(filename, edges, colors);
    }
    return {edges, colors};
}

void write_output(double elapsed_time, const vector<vector<int>>& sorted_list, const string& filename) {
    ofstream outfile(filename);
    if(!outfile.is_open()) {
        cerr << "Error: Unable to open file " << filename << endl;
        return;
    }
    
    outfile << elapsed_time << endl;
    for(const auto& node_list : sorted_list) {
        for(const auto& node : node_list) {
            outfile << node << " ";
        }
        outfile << endl;
    }
    outfile.close();
    cout << "Output written\n";
}

int main(int argc, char** argv) {
    init_mpi(argc, argv);  // MPI initialization
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get total number of processes
    
    string foldername = argv[1];
    string output_file = argv[2];
    int k = stoi(argv[3]);
    pair<vector<pair<int,int>>, map<int,int>> graph_part = read_graph_part(rank, size, foldername);

    cout << "Reading done for process" << rank << endl;

    if (rank == 0) {
        clock_t start = clock();  // Start time

        vector<vector<int>> output = degree_cen(graph_part.first, graph_part.second, k);
        
        clock_t end = clock();  // End time
        double elapsed_time = double(end - start) / CLOCKS_PER_SEC;

        write_output(elapsed_time, output, output_file);
    }
    else{
        vector<vector<int>> output1 = degree_cen(graph_part.first, graph_part.second, k);   
    }

    end_mpi();  // MPI Finalization
    
    return 0;
}

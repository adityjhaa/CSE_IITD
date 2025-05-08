#include "template.hpp"
#include <algorithm>
#include <unordered_map>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <mpi.h>

using namespace std;

void init_mpi(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
}

void end_mpi() {
    MPI_Finalize();
}

vector<vector<int>> degree_cen(vector<pair<int, int>>& partial_edge_list, map<int, int>& partial_vertex_color, int k) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<int> local_colors;
    for (const auto& pair : partial_vertex_color) {
        local_colors.push_back(pair.first);
        local_colors.push_back(pair.second);
    }

    int local_size = local_colors.size();
    vector<int> sizes(size);
    if (rank == 0) {
        sizes[0] = local_size;
        for (int src = 1; src < size; ++src) {
            MPI_Recv(&sizes[src], 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(&local_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    vector<int> displacement(size, 0);
    int total_size = 0;
    if (rank == 0) {
        for (int i = 1; i < size; ++i) {
            displacement[i] = displacement[i-1] + sizes[i-1];
        }
        total_size = displacement[size-1] + sizes[size-1];
    }

    vector<int> all_colors(total_size);
    if (rank == 0) {
        copy(local_colors.begin(), local_colors.end(), all_colors.begin());
        for (int src = 1; src < size; ++src) {
            MPI_Recv(all_colors.data() + displacement[src], sizes[src], MPI_INT, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(local_colors.data(), local_size, MPI_INT, 0, 1, MPI_COMM_WORLD);
    }

    MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    all_colors.resize(total_size);
    MPI_Bcast(all_colors.data(), total_size, MPI_INT, 0, MPI_COMM_WORLD);

    unordered_map<int, int> global_vertex_color;
    set<int> distinct_colors;
    for (int i = 0; i < total_size; i += 2) {
        global_vertex_color[all_colors[i]] = all_colors[i+1];
        distinct_colors.insert(all_colors[i+1]);
    }

    unordered_map<int, unordered_map<int, int>> local_centrality;
    for (const auto& edge : partial_edge_list) {
        int u = edge.first, v = edge.second;
        if (global_vertex_color.count(v)) {
            local_centrality[u][global_vertex_color[v]]++;
        }
        if (global_vertex_color.count(u)) {
            local_centrality[v][global_vertex_color[u]]++;
        }
    }

    vector<int> local_centrality_data;
    for (const auto& [node, color_map] : local_centrality) {
        for (const auto& [color, count] : color_map) {
            local_centrality_data.push_back(node);
            local_centrality_data.push_back(color);
            local_centrality_data.push_back(count);
        }
    }

    local_size = local_centrality_data.size();
    if (rank == 0) {
        sizes[0] = local_size;
        for (int src = 1; src < size; ++src) {
            MPI_Recv(&sizes[src], 1, MPI_INT, src, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(&local_size, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        for (int i = 1; i < size; ++i) {
            displacement[i] = displacement[i-1] + sizes[i-1];
        }
        total_size = displacement[size-1] + sizes[size-1];
    }

    vector<int> all_centrality_data(total_size);
    if (rank == 0) {
        copy(local_centrality_data.begin(), local_centrality_data.end(), all_centrality_data.begin());
        for (int src = 1; src < size; ++src) {
            MPI_Recv(all_centrality_data.data() + displacement[src], sizes[src], MPI_INT, src, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(local_centrality_data.data(), local_size, MPI_INT, 0, 3, MPI_COMM_WORLD);
    }

    MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    all_centrality_data.resize(total_size);
    MPI_Bcast(all_centrality_data.data(), total_size, MPI_INT, 0, MPI_COMM_WORLD);

    unordered_map<int, unordered_map<int, int>> global_centrality;
    for (int i = 0; i < total_size; i += 3) {
        global_centrality[all_centrality_data[i]][all_centrality_data[i+1]] += all_centrality_data[i+2];
    }

    vector<int> sorted_colors(distinct_colors.begin(), distinct_colors.end());
    sort(sorted_colors.begin(), sorted_colors.end());

    vector<vector<int>> result;
    for (int color : sorted_colors) {
        vector<pair<int, int>> nodes_with_centrality;
        for (const auto& [node, color_map] : global_centrality) {
            if (color_map.count(color)) {
                nodes_with_centrality.push_back({color_map.at(color), node});
            }
        }
        sort(nodes_with_centrality.begin(), nodes_with_centrality.end(),
            [](const pair<int, int>& a, const pair<int, int>& b) {
                return (a.first != b.first) ? (a.first > b.first) : (a.second < b.second);
            });
        vector<int> top_k_nodes;
        for (int i = 0; i < min(k, static_cast<int>(nodes_with_centrality.size())); ++i) {
            top_k_nodes.push_back(nodes_with_centrality[i].second);
        }
        result.push_back(top_k_nodes);
    }
    return (rank == 0) ? result : vector<vector<int>>();
}

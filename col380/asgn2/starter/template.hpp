#ifndef TEMPLATE_HPP
#define TEMPLATE_HPP

#include <vector>
#include <map>
#include <mpi.h>

void init_mpi(int argc, char* argv[]);

void end_mpi();

std::vector<std::vector<int>> degree_cen(std::vector<std::pair<int, int>>& partial_edge_list, 
                                         std::map<int, int>& partial_vertex_color, 
                                         int k);

#endif

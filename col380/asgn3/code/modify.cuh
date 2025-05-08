#ifndef MODIFY_CUH
#define MODIFY_CUH

#include <vector>

using namespace std;

vector<vector<int>> gen_matrix(int range, int rows, int cols);

bool check(vector<vector<vector<int>>>& upd_matrices, vector<vector<vector<int>>>& org_matrices);

// Todo
vector<vector<vector<int>>> modify(vector<vector<vector<int>>>& matrices, vector<int>& range);


#endif


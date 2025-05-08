#include <random>
#include <iostream>
#include <iomanip>
#include "modify.cuh"

#define MODIFY_ON
#define CHECK_ON

void print(vector<vector<int>>& matrix) {
  	for (int i = 0; i < matrix.size(); i++) {
    	for (int j = 0; j < matrix[0].size(); j++)
      	cout << matrix[i][j] << ' ';
  	}
  	cout << endl;
}

void check_all(){
	vector<vector<int>> test_cases = {
		{1000, 1000, 1},
		{1000, 1000, 10},
		{10000, 10000, 1},
		{10000, 10000, 10},
		{10000, 100000, 1}
	};
	
	vector<int> test_ranges = {
		1024, 
		4096, 
		100000, 
		100000000
	};	
	
	for (auto &test_case : test_cases) {
		for (int range: test_ranges) {
			int rows = test_case[0];
			int cols = test_case[1];
			int m =	 test_case[2];
			vector<vector<vector<int>>> matrices;
			matrices.reserve(m);
			for (int i = 0; i < m; i++) {
				cout << "Generating matrix: " << i + 1 << " of " << m << endl;
				matrices.push_back(gen_matrix(range, rows, cols));
			}
			vector<int> ranges(m, range);
	
			#ifdef MODIFY_ON
			cout << "Modifying matrix: row->" << rows << ", col->" << cols << ", range->" << range << ", M->" << m << endl;
			vector<vector<vector<int>>> upd_matrices = modify(matrices, ranges);
			#endif
	
			#ifdef CHECK_ON
			cout << "Checking correctness..." << endl;
			if (check(upd_matrices, matrices)) {
				cout << "Test Passed\n";
			} else {
				cout << "Test Failed\n";
				return;
			}
			#endif
		}
	}
}


int main() {

	bool test = false;
	if (test) {
		check_all();
		return 0;
	}

	int range{ 100000 }, rows{ 10000 }, cols{ 100000 };
  	cout << "Generating matrix..." << endl;
  	vector<vector<vector<int>>> matrices;
  	matrices.push_back(gen_matrix(range, rows, cols));
  	vector<int> ranges(1, range);

#ifdef MODIFY_ON
  	cout << "Modifying matrix..." << endl;
  	vector<vector<vector<int>>> upd_matrices = modify(matrices, ranges);
#endif

#ifdef CHECK_ON
	cout << "Checking correctness..." << endl;
  	if (check(upd_matrices, matrices)) cout << "Test Passed\n";
  	else cout << "Test Failed\n";
#endif
  	return 0;
}

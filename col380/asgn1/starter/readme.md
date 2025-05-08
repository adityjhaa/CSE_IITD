# OpenMP Assignment Instructions for HPC

## Module Requirements
Before running the code, you must load ONLY the following modules:
1. `module load compiler/gcc/9.1.0`
2. `module load compiler/gcc/9.1/mpich/3.3.1`

Run `module purge` before loading any modules so that just the two modules are loaded.

## Important Notes
- Do NOT modify `check.cpp` or `check.h`
- Your implementation should be written in `template.cpp` only
- Do not modify any function definitions in `template.cpp`
- In `check.cpp`, you may only modify the values of these variables:
  - n
  - m
  - b
  - k

## Testing Your Code
This is a preliminary autograder to verify if your code works correctly. The final autograder will include additional checks on top of the existing ones.

To compile and run the autograder, use these commands:
```bash
g++ -fopenmp -std=c++17 check.cpp template.cpp -o check
./check
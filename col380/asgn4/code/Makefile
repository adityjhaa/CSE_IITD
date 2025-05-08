.DEFAULT_GOAL := all

MPICXX = mpic++

CXXFLAGS = -O3 -std=c++17 -fopenmp

TARGET = a4
NUM_PROCS = 4
FOLDER = TestCases/test

all: $(TARGET)

$(TARGET): matrix.o multiply.o main.o
	$(MPICXX) $(CXXFLAGS) -o $@ $^ -lcudart
	@rm -f *.o

main.o: main.cpp matrix.hpp
	$(MPICXX) -O3 -std=c++17 -fopenmp -c main.cpp -o main.o -lcudart

matrix.o: matrix.cpp matrix.hpp
	$(MPICXX) -O3 -std=c++17 -c matrix.cpp -o matrix.o

multiply.o: multiply.cu matrix.hpp
	nvcc -arch=sm_35 -c multiply.cu -Xcompiler -fopenmp -o multiply.o

clean:
	@rm -f $(TARGET) *.o
	@rm -f *.txt
	@rm -f matrix


.PHONY: all clean

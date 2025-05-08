CXX = g++
CXXFLAGS = -std=c++17 -O3

TARGET = main

SRCS = main.cpp io.cpp matrix.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	@$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)
	@rm -f $(OBJS)

%.o: %.cpp
	@$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	@rm -f $(OBJS) $(TARGET)
	@rm -f input/* output/*

.PHONY: all clean


CXX = g++
# CXXFLAGS = -std=c++17 -Wall -Wextra -pthread
LDFLAGS = -pthread

all: server client

build: server.cpp client.cpp
	@$(CXX) -o server server.cpp $(LDFLAGS)
	@$(CXX) -o client client.cpp $(LDFLAGS)

server: build
	@./server 1

client: client.cpp
	@./client

run-fifo: build
	@./server 1 & sleep 1 && ./client && killall server

run-rr: build
	@./server 2 & sleep 1 && ./client && killall server

run: build
	@./server 3 & sleep 1 && ./client 1 && killall server

plot:
	@python3 plot.py
	@rm -f server client
	@rm -f client_*txt
	@rm -f *csv

fairness: build
	@./server 4 & sleep 1 && ./client 2 && killall server

clean:
	@rm -f server client
	@rm -f client_*txt
	@rm -f *.csv
	@rm -f plot.png


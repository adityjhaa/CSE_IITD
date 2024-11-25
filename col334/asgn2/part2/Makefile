CXX = g++
SERVER_SRC = server.cpp
CLIENT_SRC = client.cpp
SERVER_BIN = server
CLIENT_BIN = client

.PHONY: all build server client run clean

all: build

build: $(SERVER_SRC) $(CLIENT_SRC)
	@$(CXX) -o $(SERVER_BIN) $(SERVER_SRC)
	@$(CXX) -o $(CLIENT_BIN) $(CLIENT_SRC)

server: $(SERVER_SRC)
	@$(CXX) -o $(SERVER_BIN) $(SERVER_SRC)
	@./$(SERVER_BIN)  # Run server in the background

client: $(CLIENT_SRC)
	@$(CXX) -o $(CLIENT_BIN) $(CLIENT_SRC)
	@./$(CLIENT_BIN)  # Run the client

run: build
	@./$(SERVER_BIN) &  # Run server in the background
	@sleep 1           # Give the server a moment to start
	@./$(CLIENT_BIN)    # Run the client
	@killall server     # Kill the server

plot: 
	@python3 plot.py

clean:
	@rm -f $(SERVER_BIN) $(CLIENT_BIN)
	@rm -f client_*.txt

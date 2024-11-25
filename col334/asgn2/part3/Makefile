CXX = g++
SERVER_SRC = server.cpp
CLIENT_SRC = client.cpp
SERVER_BIN = server
CLIENT_BIN = client

.PHONY: all build server client run-cscd run-beb run-aloha clean

all: build

build: $(SERVER_SRC) $(CLIENT_SRC)
	@$(CXX) -o $(SERVER_BIN) $(SERVER_SRC)
	@$(CXX) -o $(CLIENT_BIN) $(CLIENT_SRC)

server: $(SERVER_SRC)
	@$(CXX) -o $(SERVER_BIN) $(SERVER_SRC)

client: $(CLIENT_SRC)
	@$(CXX) -o $(CLIENT_BIN) $(CLIENT_SRC)

run-cscd: build
	@./$(SERVER_BIN) &  
	@sleep 1           
	@./$(CLIENT_BIN) cscd  
	@killall server

run-beb: build
	@./$(SERVER_BIN) &  
	@sleep 1           
	@./$(CLIENT_BIN) beb  
	@killall server

run-aloha: build
	@./$(SERVER_BIN) &  
	@sleep 1           
	@./$(CLIENT_BIN) slotted-aloha  
	@killall server

plot: 
	@python3 plot.py

clean:
	@rm -f $(SERVER_BIN) $(CLIENT_BIN)
	@rm -f client_*.txt

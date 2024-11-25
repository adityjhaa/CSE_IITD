CXX = g++
SERVER_SRC = server.cpp
CLIENT_SRC = client.cpp
SERVER_BIN = server
CLIENT_BIN = client

.PHONY: all build server client run plot clean

all: build


build: $(SERVER_SRC) $(CLIENT_SRC)
	@$(CXX) -o $(SERVER_BIN) $(SERVER_SRC)
	@$(CXX) -o $(CLIENT_BIN) $(CLIENT_SRC)

server: $(SERVER_SRC)
	@$(CXX) -o $(SERVER_BIN) $(SERVER_SRC)
	@./$(SERVER_BIN)

client: $(CLIENT_SRC)
	@$(CXX) -o $(CLIENT_BIN) $(CLIENT_SRC)
	@./$(CLIENT_BIN)


run: build
	@./$(SERVER_BIN) &
	@sleep 1
	@./$(CLIENT_BIN)

plot:
	python3 plot.py

clean:
	@rm -f $(SERVER_BIN) $(CLIENT_BIN)
  @rm -f client*txt
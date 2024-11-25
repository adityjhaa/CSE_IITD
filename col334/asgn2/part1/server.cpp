#include "helper.hpp"

#include <csignal>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#define BUFFER_SIZE 1024
using namespace std;

void handle_client(int clientSocket, const int k, const int p, const vector<string> &words);
int send_words(int clientSocket, const int offset, const int k, const int p, const vector<string> &words);

int main() {
    // parse and find the values from the json file
    string input_file, server_ip;
    int server_port, k, p;
    parse_server(server_ip, server_port, k, p, input_file);

    vector<string> words = load_words(input_file);

    int serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket == -1) {
        cerr << "Error creating socket" << endl;
        return 1;
    }
    
    int opt = 1;
    if (setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        std::cerr << "Error setting SO_REUSEADDR option: " << strerror(errno) << std::endl;
        return 1;
    }
 
    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = inet_addr(server_ip.c_str());
    serverAddr.sin_port = htons(server_port);

    if (bind(serverSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        cerr << "Error binding socket: " << strerror(errno) << endl;
        close(serverSocket);
        return 1;
    }

    if (listen(serverSocket, 1) < 0) {
        cerr << "Error listening on socket" << endl;
        return 1;
    }

    sockaddr_in clientAddr;
    socklen_t clientAddrLen = sizeof(clientAddr);
    int clientSocket = accept(serverSocket, (struct sockaddr*)&clientAddr, &clientAddrLen);

    if (clientSocket < 0) {
        cerr << "Error accepting connection" << endl;
    }

    handle_client(clientSocket, k, p, words);

    close(serverSocket);
    return 0;
}

void handle_client(int clientSocket, const int k, const int p, const vector<string> &words) {
    char buffer[BUFFER_SIZE];
    int offset = 0;
    int total = 0;

    while (true) {
        memset(buffer, 0, BUFFER_SIZE);
        int bytes_received = recv(clientSocket, buffer, BUFFER_SIZE, 0);
        
        if (bytes_received <= 0) {
            cerr << "Client disconnected" << endl;
            break;
        }
        
        sscanf(buffer, "%d\n", &offset);

        if (offset >= words.size()) {
            string response = "$$\n";
            send(clientSocket, response.c_str(), response.size(), 0);
            return;
        }

        total += send_words(clientSocket, offset, k, p, words);
    }

    close(clientSocket);
}

int send_words(int clientSocket, const int offset, const int k, const int p, const vector<string> &words) {
    int words_to_send = min(k, static_cast<int>(words.size()) - offset);
    bool last = (words.size() - offset) <= k;

    int temp = 0;
    string packets = "";
    for (int i = 0; i < words_to_send; i += p) {
        string packet;
        int words_in_packet = min(p, words_to_send - i);

        int j = 0;
        for (; j < words_in_packet; j++) {
            packet += words[offset + i + j];
            temp++;
            if (j < words_in_packet - 1)
                packet += ',';
        }

        if (last && (i + j >= words.size()))
            packet += ",EOF";
        packet += '\n';

        packets += packet;
    }

    ssize_t sent_bytes = send(clientSocket, packets.c_str(), packets.size(), 0);    
    if (sent_bytes < 0) {
        cerr << "Error sending data: " << strerror(errno) << endl;
        return -1;
    }
    return temp;
}


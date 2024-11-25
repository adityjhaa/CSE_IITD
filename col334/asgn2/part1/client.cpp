#include "helper.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <pthread.h>
#include <sstream>
#include <string>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

#define BUFFER_SIZE 1024
using namespace std;

void process_packet(const string &packet, map<string, int> &map);
void output(const map<string, int> &map);

int main() {
    // parse and find the values from the json file
    string server_ip;
    int server_port, k, p;
    parse_client(server_ip, server_port, k, p);

    int clientSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (clientSocket == -1) {
        cerr << "Error creating socket" << endl;
        return 1;
    }

    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(server_port);

   if (inet_pton(AF_INET, server_ip.c_str(), &serverAddr.sin_addr) <= 0) {
        cerr << "Invalid address" << endl;
        return 1;
    }

    if (connect(clientSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        cerr << "Connection failed" << endl;
        return 1;
    }
    
    // char buffer[BUFFER_SIZE] = {0};
    int offset{};
    string packet;
    bool eof = false;

    map<string, int> map;

    while (!eof) {
        string offsetStr = to_string(offset) + "\n";
        send(clientSocket, offsetStr.c_str(), offsetStr.length(), 0);

        char buffer[BUFFER_SIZE] = {0};
        
        int bytesRead = recv(clientSocket, buffer, BUFFER_SIZE, 0);
        if (bytesRead > 0) {
            buffer[bytesRead] = '\0';
            packet = string(buffer);
                
            if (packet == "$$\n") {break;}
            size_t eof_pos = packet.find("EOF");
            if (eof_pos != string::npos) {
                eof = true;
                packet = packet.substr(0, eof_pos);
            }

            process_packet(packet, map);
            packet.clear();
        } else {
            cerr << "Error receiving data: " << strerror(errno) << endl;
            break;
        }
        offset += k;
    }
    close(clientSocket);

    output(map);

    return 0;
}

void process_packet(const string &packet, map<string, int> &map) {
    istringstream iss(packet);
    string line;
    while (getline(iss, line)) {
        istringstream isl(line);
        string word;
        while (getline(isl, word, ',')) {
            if (!word.empty())
                map[word]++;
        }
    }
}

void output(const map<string, int> &word_count) {
    ofstream output_file("client_1.txt");

    if (!output_file.is_open()) {
        cerr << "Could not open output file." << endl;
        return;
    }

    auto it = word_count.begin();
    while (it != word_count.end()) {
        output_file << it->first << ", " << it->second;
        ++it;
        if (it != word_count.end()) {
            output_file << endl;
        }
    }

    output_file.close();
}

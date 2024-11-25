#pragma once

#include "json.hpp"

#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>


#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

using namespace std;
using nlohmann::json;

#define BUFFER_SIZE 1024

// basic stuff start
json read_config(const string &config_file) {
    ifstream file(config_file);
    if (!file.is_open()) {
        cerr << "Could not open config file: " << config_file << endl;
        exit(1);
    }

    json config;
    file >> config;
    return config;
}

vector<string> load_words(const string &file_name) {
    ifstream file(file_name);

    vector<string> words;

    string word;
    while (getline(file, word, ','))    
        words.push_back(word);

    return words;
}
// basic stuff end

// client for the server side
struct ClientRequest {
    int clientSocket;
    int offset;
    chrono::steady_clock::time_point arrivalTime;
};

struct Clients {
    int offset;
    map<string, int> word_count;
};

double calculateJainIndex(const vector<double>& values) {
    double sum = accumulate(values.begin(), values.end(), 0.0);
    double sum_of_squares = inner_product(values.begin(), values.end(), values.begin(), 0.0);
    double n = values.size();
    return (sum * sum) / (n * sum_of_squares);
}


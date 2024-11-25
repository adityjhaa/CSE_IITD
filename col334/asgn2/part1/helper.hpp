#pragma once

#include <vector>
#include <string>
#include <fstream>

using namespace std;

void parse_server(string &server_ip, int &server_port, int &k, int &p, string &input_file) {
    ifstream config_file("config.json");
    string line;

    while (getline(config_file, line)) {
        if (line.find("\"server_ip\"") != string::npos) {
            server_ip = line.substr(line.find(":") + 2);
            server_ip.pop_back();
            server_ip.erase(0, server_ip.find_first_not_of("\""));
            server_ip.erase(server_ip.find_last_not_of("\"") + 1);
        } else if (line.find("\"server_port\"") != string::npos) {
            server_port = stoi(line.substr(line.find(":") + 1));
        } else if (line.find("\"k\"") != string::npos) {
            k = stoi(line.substr(line.find(":") + 1));
        } else if (line.find("\"p\"") != string::npos) {
            p = stoi(line.substr(line.find(":") + 1));
        } else if (line.find("\"input_file\"") != string::npos) {
            input_file = line.substr(line.find(":") + 2);
            input_file.pop_back();
            input_file.erase(0, input_file.find_first_not_of("\""));
            input_file.erase(input_file.find_last_not_of("\"") + 1);
        }
    }
}

void parse_client(string &server_ip, int &server_port, int &k, int &p) {
    ifstream config_file("config.json");
    string line;

    while (getline(config_file, line)) {
        if (line.find("\"server_ip\"") != string::npos) {
            server_ip = line.substr(line.find(":") + 2);
            server_ip.pop_back();
            server_ip.erase(0, server_ip.find_first_not_of("\""));
            server_ip.erase(server_ip.find_last_not_of("\"") + 1);
        } else if (line.find("\"server_port\"") != string::npos) {
            server_port = stoi(line.substr(line.find(":") + 1));
        } else if (line.find("\"k\"") != string::npos) {
            k = stoi(line.substr(line.find(":") + 1)); 
        } else if (line.find("\"p\"") != string::npos) {
            p = stoi(line.substr(line.find(":") + 1));
        }
    }
}

vector<string> load_words(const string &file_name) {
    ifstream file(file_name);

    vector<string> words;

    string word;
    while (getline(file, word, ','))
        words.push_back(word);

    return words;
}

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <vector>
#include <map>
#include <sstream>
#include <chrono>
#include <pthread.h>
#include "json.hpp"

using json = nlohmann::json;
using namespace std;

#define BUFFER_SIZE 1024

struct ClientArgs
{
    int client_id;
    string server_ip;
    int port;
    int num_words_per_req;
    std::chrono::time_point<std::chrono::system_clock> start_time;
    std::chrono::time_point<std::chrono::system_clock> end_time;
};

void *client_thread(void *arg)
{
    ClientArgs *args = (ClientArgs *)arg;
    int client_id = args->client_id;
    string server_ip = args->server_ip;
    int port = args->port;
    int num_words_per_req = args->num_words_per_req;

    // Record the start time for the client
    args->start_time = chrono::system_clock::now();

    int server_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (server_sock < 0)
    {
        cerr << "Socket creation failed for client " << client_id << endl;
        return nullptr;
    }

    struct sockaddr_in server_address;
    server_address.sin_family = AF_INET;
    server_address.sin_port = htons(port);

    if (inet_pton(AF_INET, server_ip.c_str(), &server_address.sin_addr) <= 0)
    {
        cerr << "Invalid server IP address for client " << client_id << endl;
        return nullptr;
    }

    if (connect(server_sock, (struct sockaddr *)&server_address, sizeof(server_address)) < 0)
    {
        cerr << "Connection to server failed for client " << client_id << endl;
        return nullptr;
    }

    int offset = 0;
    map<string, int> word_count;

    while (true)
    {
        string request = to_string(offset) + "\\n";
        send(server_sock, request.c_str(), request.size(), 0);

        char buffer[BUFFER_SIZE + 1];
        int bytes_received = recv(server_sock, buffer, BUFFER_SIZE, 0);
        if (bytes_received <= 0)
        {
            cerr << "Error receiving data or connection closed by server for client " << client_id << endl;
            break;
        }
        buffer[bytes_received] = '\0';

        string response(buffer);

        if (response == "$$\\n")
        {
            cerr << "Client " << client_id << " received EOF from the server." << endl;
            close(server_sock);
            break;
        }

        stringstream ss(response);
        usleep(10);
        string word;
        while (getline(ss, word, ','))
        {
            if (word.find("EOF") != string::npos)
            {
                word.erase(word.find("EOF"));
            }
            if (word.find("\\n") != string::npos)
            {
                word = word.substr(2, word.size() - 2);
            }
            if (!word.empty() && word != "\\n" && word != "EOF\\n")
            {
                word_count[word]++;
            }
        }

        if (response.find("EOF") != string::npos)
        {
            response.erase(response.find("EOF"));
            break;
        }

        offset += num_words_per_req;
    }

    ofstream output_file("client_" + to_string(client_id + 1) + ".txt");
    if (!output_file.is_open())
    {
        cerr << "Could not open output file for client " << client_id << endl;
        return nullptr;
    }

    auto it = word_count.begin();
    while (it != word_count.end())
    {
        output_file << it->first << ", " << it->second;
        ++it;
        if (it != word_count.end())
            output_file << endl;
    }

    output_file.close();

    // Record the end time for the client
    args->end_time = chrono::system_clock::now();

    close(server_sock);
    return nullptr;
}

json read_config(const string &config_file)
{
    ifstream file(config_file);
    if (!file.is_open())
    {
        cerr << "Could not open config file: " << config_file << endl;
        exit(1);
    }
    json config;
    file >> config;
    return config;
}

int main()
{
    json config = read_config("config.json");
    string server_ip = config["server_ip"];
    int port = config["server_port"];
    int num_clients = config["num_clients"];
    int num_words_per_req = config["k"];

    vector<pthread_t> threads(num_clients);
    vector<ClientArgs *> client_args(num_clients); // Store ClientArgs for each thread
    vector<double> client_times(num_clients);      // Store completion times for each client

    for (int i = 0; i < num_clients; ++i)
    {
        client_args[i] = new ClientArgs{i, server_ip, port, num_words_per_req, chrono::system_clock::now(), chrono::system_clock::now()};
        pthread_create(&threads[i], nullptr, client_thread, client_args[i]);
    }

    for (int i = 0; i < num_clients; ++i)
    {
        pthread_join(threads[i], nullptr); // Wait for all threads to finish

        // Calculate the completion time for each client
        chrono::duration<double> duration = client_args[i]->end_time - client_args[i]->start_time;
        client_times[i] = duration.count();

        // Clean up allocated memory
        delete client_args[i];
    }

    // Calculate the average completion time
    double total_time = 0;
    for (double time : client_times)
    {
        total_time += time;
    }
    double avg_time = total_time / num_clients;

    // Write the average time to avg_time.txt
    ofstream avg_time_file("avg_time.txt");
    if (avg_time_file.is_open())
    {
        avg_time_file << avg_time << endl;
        avg_time_file.close();
    }
    else
    {
        cerr << "Could not open avg_time.txt for writing." << endl;
    }

    return 0;
}

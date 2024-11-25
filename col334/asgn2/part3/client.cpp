#include <iostream>
#include <fstream>
#include <cstring>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <chrono>
#include <pthread.h>
#include <random>
#include <vector>
#include "json.hpp"

using json = nlohmann::json;
using namespace std;

#define BUFFER_SIZE 1024

struct ClientConfig
{
    string server_ip;
    int server_port;
    int client_id;
    int p;
    int num_words_per_req;
    int T; 
    string protocol;
    int num_clients;
    chrono::time_point<chrono::system_clock> start_time;
    chrono::time_point<chrono::system_clock> end_time;
};

random_device rd;
mt19937 rng(rd());

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

// Function to simulate Slotted Aloha
void slotted_aloha(int slot_time, double p)
{
    uniform_real_distribution<double> dist(0, 1);
    if (dist(rng) > p)
    {
        return;
    }
    usleep(slot_time * 1000);
}

// Function to simulate Binary Exponential Backoff (BEB)
void binary_exponential_backoff(int attempt)
{
    int backoff_time = pow(2, attempt); // in milliseconds
    uniform_int_distribution<int> dist(0, backoff_time);
    usleep(1000 * (dist(rng)));
}

// Function to simulate Carrier Sensing with BEB
bool carrier_sensing_with_beb(int client_sock)
{
    string sense_req = "BUSY?\\n";
    send(client_sock, sense_req.c_str(), sense_req.size(), 0);

    char buffer[BUFFER_SIZE] = {0};
    memset(buffer, 0, BUFFER_SIZE);
    recv(client_sock, buffer, BUFFER_SIZE, 0);

    if (strcmp(buffer, "BUSY\\n") == 0)
    {
        return true;
    }
    return false;
}

// Function to handle sending requests and receiving responses from the server
void *client_thread(void *arg)
{
    ClientConfig *config = (ClientConfig *)arg;
    config->start_time = chrono::system_clock::now();  // Record start time

    int client_id = config->client_id;
    int num_words_per_req = config->num_words_per_req;
    string protocol = config->protocol;
    string server_ip = config->server_ip;
    int server_port = config->server_port;
    int T = config->T;
    double prob = 1/(double)config->num_clients;

    // Create a socket
    int server_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (server_sock < 0)
    {
        cerr << "Socket creation failed" << endl;
        return nullptr;
    }

    struct sockaddr_in server_address;
    server_address.sin_family = AF_INET;
    server_address.sin_port = htons(server_port);

    if (inet_pton(AF_INET, server_ip.c_str(), &server_address.sin_addr) <= 0)
    {
        cerr << "Invalid address or address not supported" << endl;
        return nullptr;
    }

    // Connect to the server
    if (connect(server_sock, (struct sockaddr *)&server_address, sizeof(server_address)) < 0)
    {
        cerr << "Connection to server failed" << endl;
        return nullptr;
    }

    int attempt = 0;
    int offset = 0;
    map<string, int> word_count;

    while (true)
    {
        if (protocol == "slotted-aloha")
        {
            slotted_aloha(T, prob);
        }
        else if (protocol == "beb")
        {
            if (attempt > 0)
            {
                binary_exponential_backoff(attempt);
            }
        }
        else if (protocol == "cscd")
        {
            if (carrier_sensing_with_beb(server_sock))
            {
                binary_exponential_backoff(attempt);
                attempt++;
                continue;
            }
        }

        string request = to_string(offset) + "\\n";
        int bytes_sent = send(server_sock, request.c_str(), request.size(), 0);

        char buffer[BUFFER_SIZE] = {0};
        int bytes_received = recv(server_sock, buffer, BUFFER_SIZE, 0);
        if (bytes_received <= 0)
        {
            close(server_sock);
            break;
        }

        buffer[bytes_received] = '\0';
        string response(buffer);

        if (response == "$$\\n")
        {
            close(server_sock);
            break;
        }

        stringstream ss(response);
        string word;
        bool got_huh = false;

        while (getline(ss, word, ','))
        {
            if (word.find("HUH!") != string::npos)
            {
                got_huh = true;
                attempt++;
                break;
            }
        }

        if (got_huh)
        {
            continue;
        }

        ss.clear();
        ss.str(response);
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
        usleep(10);
    }
    ofstream output_file("client_" + to_string(client_id) + ".txt");
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
    config->end_time = chrono::system_clock::now();  // Record end time
    close(server_sock);
    return nullptr;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " <protocol: slotted-aloha | beb | cscd>" << endl;
        return 1;
    }

    string protocol = argv[1]; // Protocol is provided via command-line arguments

    // Reading configuration from config.json
    json config = read_config("config.json");

    string server_ip = config["server_ip"];
    int server_port = config["server_port"];
    int num_clients = config["num_clients"];
    int p = config["p"];
    int num_words_per_req = config["k"];
    int T = config["T"]; // Slot time from config

    pthread_t threads[num_clients];
    ClientConfig client_configs[num_clients];

    // Creating and launching client threads
    for (int i = 0; i < num_clients; ++i)
    {
        client_configs[i] = {server_ip, server_port, i+1, p, num_words_per_req, T, protocol, num_clients, chrono::system_clock::now(), chrono::system_clock::now()}; // Initialize client configuration
        pthread_create(&threads[i], nullptr, client_thread, (void *)&client_configs[i]);
    }

    // Joining threads after they complete
    for (int i = 0; i < num_clients; ++i)
    {
        pthread_join(threads[i], nullptr);
    }

    // Calculating and logging the average completion time
    double total_time = 0.0;
    for (int i = 0; i < num_clients; ++i)
    {
        chrono::duration<double> elapsed_time = client_configs[i].end_time - client_configs[i].start_time;
        total_time += elapsed_time.count();
    }
    double avg_time = total_time / num_clients;

    // Log to avg_time.txt
    ofstream avg_time_file("avg_time.txt");
    if (avg_time_file.is_open())
    {
        avg_time_file << avg_time << endl;
        avg_time_file.close();
    }
    else
    {
        cerr << "Unable to open avg_time.txt" << endl;
    }

    return 0;
}

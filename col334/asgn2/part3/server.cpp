#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <vector>
#include <sstream>
#include <pthread.h>
#include <chrono>
#include "json.hpp"

using json = nlohmann::json;
using namespace std;
using namespace chrono;

#define BUFFER_SIZE 1024

vector<int> active_clients;
bool collision_detected = false;

struct ClientData
{
    int client_sock;
    const vector<string> *words;
    int p;
    int k;
};

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

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

vector<string> tokenize_mmap_file(const char *file_data, size_t file_size)
{
    vector<string> words;
    stringstream ss;

    for (size_t i = 0; i < file_size; ++i)
    {
        if (file_data[i] == ',')
        {
            words.push_back(ss.str());
            ss.str("");
        }
        else
        {
            ss << file_data[i];
        }
    }
    if (!ss.str().empty())
    {
        words.push_back(ss.str());
    }

    return words;
}

void send_huh(int client_sock)
{
    string huh = "HUH!\\n";
    send(client_sock, huh.c_str(), huh.size(), 0);
}

void handle_collision()
{
    pthread_mutex_lock(&mutex);
    for (int client_sock : active_clients)
    {
        send_huh(client_sock);
    }
    collision_detected = false;
    active_clients.clear();
    pthread_mutex_unlock(&mutex);
}

void *handle_client(void *arg)
{
    ClientData *data = (ClientData *)arg;
    int client_sock = data->client_sock;
    const vector<string> &words = *data->words;
    int p = data->p;
    int k = data->k;

    char buffer[BUFFER_SIZE];
    int offset = 0;

    while (true)
    {
        memset(buffer, 0, BUFFER_SIZE);
        int read_size = recv(client_sock, buffer, BUFFER_SIZE, 0);

        if (read_size <= 0)
        {
            cerr << "Client disconnected or error receiving data" << endl;
            close(client_sock);
            break;
        }

        if (strcmp(buffer, "BUSY?\\n") == 0)
        {
            string response = active_clients.size() >= 1 ? "BUSY\\n" : "IDLE\\n";
            send(client_sock, response.c_str(), response.size(), 0);
            continue;
        }

        sscanf(buffer, "%d\n", &offset);
        active_clients.push_back(client_sock);
        if (active_clients.size() > 1)
        {
            collision_detected = true;
            handle_collision();
            pthread_mutex_unlock(&mutex);
            continue;
        }

        pthread_mutex_lock(&mutex);
        if (offset < 0 || offset >= words.size())
        {
            string response = "$$\\n";
            send(client_sock, response.c_str(), response.size(), 0);
            active_clients.erase(remove(active_clients.begin(), active_clients.end(), client_sock), active_clients.end());
            pthread_mutex_unlock(&mutex);
            continue;
        }

        string response_batch;
        int total_words_sent = 0;
        while (total_words_sent < k && offset + total_words_sent < words.size())
        {
            int words_to_send = min(p, static_cast<int>(words.size() - offset - total_words_sent));
            for (int i = 0; i < words_to_send; ++i)
            {
                response_batch += words[offset + total_words_sent] + ",";
            }
            if (offset + total_words_sent >= words.size())
            {
                response_batch += "EOF";
            }
            total_words_sent += words_to_send;
            response_batch += "\\n";
        }
        send(client_sock, response_batch.c_str(), response_batch.size(), 0);

        active_clients.erase(remove(active_clients.begin(), active_clients.end(), client_sock), active_clients.end());
        pthread_mutex_unlock(&mutex);
    }

    close(client_sock);
    delete data;
    return nullptr;
}

int main(int argc, char *argv[])
{
    json config = read_config("config.json");

    string server_ip = config["server_ip"];
    int server_port = config["server_port"];
    string input_file = config["input_file"];
    int p = config["p"];
    int k = config["k"];

    int fd = open(input_file.c_str(), O_RDONLY);
    if (fd == -1)
    {
        cerr << "Error opening file: " << input_file << endl;
        return 1;
    }

    size_t file_size = lseek(fd, 0, SEEK_END);
    const char *file_data = (char *)mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (file_data == MAP_FAILED)
    {
        cerr << "Error mapping file: " << input_file << endl;
        return 1;
    }

    vector<string> words = tokenize_mmap_file(file_data, file_size);
    munmap((void *)file_data, file_size);
    close(fd);

    int server_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (server_sock < 0)
    {
        cerr << "Socket creation failed" << endl;
        return 1;
    }

    struct sockaddr_in server_address;
    server_address.sin_family = AF_INET;
    server_address.sin_port = htons(server_port);
    server_address.sin_addr.s_addr = INADDR_ANY;

    if (bind(server_sock, (struct sockaddr *)&server_address, sizeof(server_address)) < 0)
    {
        cerr << "Binding failed" << endl;
        return 1;
    }

    if (listen(server_sock, 10) < 0)
    {
        cerr << "Listening failed" << endl;
        return 1;
    }

    while (true)
    {
        struct sockaddr_in client_address;
        socklen_t client_addr_len = sizeof(client_address);
        int client_sock = accept(server_sock, (struct sockaddr *)&client_address, &client_addr_len);

        if (client_sock < 0)
        {
            cerr << "Failed to accept client connection" << endl;
            continue;
        }

        ClientData *data = new ClientData{client_sock, &words, p, k};
        pthread_t client_thread;
        pthread_create(&client_thread, nullptr, handle_client, (void *)data);
        pthread_detach(client_thread); // Automatically reclaim thread resources after completion
    }

    close(server_sock);
    return 0;
}

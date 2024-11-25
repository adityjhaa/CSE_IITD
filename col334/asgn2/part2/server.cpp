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
#include "json.hpp"

using json = nlohmann::json;
using namespace std;

#define BUFFER_SIZE 1024

// Structure to hold client data
struct ClientData
{
    int client_sock;
    const vector<string> *words;
    int p;
    int k;
};

// Function to read the configuration from config.json
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
        int bytes_received = recv(client_sock, buffer, BUFFER_SIZE, 0);

        if (bytes_received <= 0)
        {
            cerr << "Client disconnected" << endl;
            break;
        }
        sscanf(buffer, "%d\n", &offset);
        cout << offset << endl;
        if (offset < 0 || offset >= words.size())
        {
            string response = "$$\\n";
            send(client_sock, response.c_str(), response.size(), 0);
            continue;
        }

        int total_words_sent = 0;
        bool eof_sent = false;

        while (total_words_sent < k && offset + total_words_sent < words.size())
        {
            string response_batch;
            int words_to_send = min(p, static_cast<int>(words.size() - offset - total_words_sent));

            for (int i = 0; i < words_to_send; ++i)
            {
                response_batch += words[offset + total_words_sent + i] + ',';
            }

            if (offset + total_words_sent + words_to_send >= words.size())
            {
                eof_sent = true;
                response_batch += "EOF";
            }

            response_batch += "\\n";
            send(client_sock, response_batch.c_str(), response_batch.size(), 0);
            total_words_sent += words_to_send;
        }
    }

    close(client_sock);
    delete data; // Free the allocated memory
    return nullptr;
}

int main()
{
    json config = read_config("config.json");
    int port = config["server_port"];
    string word_file = config["input_file"];
    int p = config["p"];
    int k = config["k"];

    int fd = open(word_file.c_str(), O_RDONLY);
    if (fd < 0)
    {
        cerr << "Could not open word file: " << word_file << endl;
        return 1;
    }

    size_t file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    char *file_data = (char *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (file_data == MAP_FAILED)
    {
        cerr << "Memory mapping failed" << endl;
        return 1;
    }

    vector<string> words = tokenize_mmap_file(file_data, file_size);
    close(fd);

    int server_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (server_sock < 0)
    {
        cerr << "Socket creation failed" << endl;
        return 1;
    }

    struct sockaddr_in server_address;
    server_address.sin_family = AF_INET;
    server_address.sin_port = htons(port);
    server_address.sin_addr.s_addr = INADDR_ANY;

    if (bind(server_sock, (struct sockaddr *)&server_address, sizeof(server_address)) < 0)
    {
        cerr << "Bind failed" << endl;
        return 1;
    }

    if (listen(server_sock, 3) < 0)
    {
        cerr << "Listen failed" << endl;
        return 1;
    }

    cout << "Server listening on port " << port << endl;

    while (true)
    {
        int client_sock = accept(server_sock, NULL, NULL);
        if (client_sock < 0)
        {
            cerr << "Client connection failed" << endl;
            continue;
        }

        ClientData *data = new ClientData{client_sock, &words, p, k};
        pthread_t thread;
        pthread_create(&thread, nullptr, handle_client, data);
        pthread_detach(thread); // Allow the thread to clean up after itself
    }

    close(server_sock);
    return 0;
}

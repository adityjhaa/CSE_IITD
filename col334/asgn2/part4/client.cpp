#include "helper.hpp"
#include <atomic>
#include <fstream>
#include <thread>

class Client {
public: // Member functions (public)
    Client(const json& config, bool flg= false)
        : server_ip(config["server_ip"]),
          server_port(config["server_port"]),
          num_clients(config["num_clients"]),
          fair(flg) {
        if (fair) num_clients = 10;
        clients.resize(num_clients);
        for (int i = 0; i < num_clients; i++) {
            clients[i] = (struct Clients){0, (map<string, int>){}};
        }
    }

    void run(int &avg) {
        vector<thread> threads;
        for (int i = 0; i < num_clients; ++i) {
            threads.emplace_back(&Client::sendRequest, this, clients[i].offset, i);
        }

        for (auto& t : threads) {
            t.join();
        }

        avg = getStats();
        for (int i = 0; i < num_clients; i++) {
            ofstream output_file("client_" + to_string(i+1) + ".txt");
            if (!output_file.is_open()) {
                cerr << "Could not open output file." << endl;
                return;
            }

            auto it = clients[i].word_count.begin();
            while (it != clients[i].word_count.end()) {
                output_file << it->first << ", " << it->second;
                ++it;
                if (it != clients[i].word_count.end()) {
                    output_file << endl;
                }
            }

            output_file.close();
        }
    }

    void run_fair(int &avg, double &jains_index) {
        vector<thread> threads;
        for (int i = 0; i < num_clients; ++i) {
            threads.emplace_back(&Client::sendRequestfair, this, clients[i].offset, i);
        }

        for (auto& t : threads) {
            t.join();
        }

        for (int i = 0; i < num_clients; i++) {
            ofstream output_file("client_" + to_string(i+1) + ".txt");
            if (!output_file.is_open()) {
                cerr << "Could not open output file." << endl;
                return;
            }

            auto it = clients[i].word_count.begin();
            while (it != clients[i].word_count.end()) {
                output_file << it->first << ", " << it->second;
                ++it;
                if (it != clients[i].word_count.end()) {
                    output_file << endl;
                }
            }

            output_file.close();
        }

        avg = getStats();
        vector<double> times;
        for (int i = 0; i < completion_times.size(); i++)
            times.push_back(static_cast<double>(completion_times[i].count()));

        jains_index = calculateJainIndex(times);
    }

private: // Member functions (private)
    void sendRequest(int initial_offset, int index) {
        int clientSocket = socket(AF_INET, SOCK_STREAM, 0);
        if (clientSocket == -1) {
            cerr << "Error creating socket" << endl;
            return;
        }

        sockaddr_in serverAddr;
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_addr.s_addr = inet_addr(server_ip.c_str());
        serverAddr.sin_port = htons(server_port);

        if (connect(clientSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
            cerr << "Error connecting to server" << endl;
            close(clientSocket);
            return;
        }

        auto start_time = chrono::steady_clock::now();

        int offset = initial_offset;
        bool eof_received = false;
        char buffer[BUFFER_SIZE];

        while (not eof_received) {
            string request = to_string(offset) + "\n";
            send(clientSocket, request.c_str(), request.length(), 0);

            int bytesRead = recv(clientSocket, buffer, BUFFER_SIZE - 1, 0);
            if (bytesRead <= 0) {
                cerr << "Server disconnected\n";
                break;
            }
            buffer[bytesRead] = '\0';

            string response(buffer);
            if (response == "$$\n") {
                break;
            }

            istringstream iss(response);
            string line;
            while (getline(iss, line, '\n')) {
                istringstream il(line);
                string word;
                while (getline(il, word, ',')) {
                    if (word == "EOF") {
                        eof_received = true;
                        cout << "recieved EOF\n";
                        break;
                    }
                    if (!word.empty()) {
                        clients[index].word_count[word]++;
                        offset++;
                    }
                }
                if (eof_received) break;
            }

        }

        close(clientSocket);

        auto end_time = chrono::steady_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);

        lock_guard<mutex> lock(completion_times_mutex);
        completion_times.push_back(duration);
    }

    void sendRequestfair(int initial_offset, int index) {
        if (index != 0) {
            sendRequest(initial_offset, index);
        } else {
            vector <thread> threads;
            atomic<bool> completed(false);
            for (int i = 0; i < 5; ++i) {
                // 5 concurrent requests with same starting offset, whichever gets the complete file first is considered and the rest are discarded.
                threads.emplace_back([this, initial_offset, index, &completed]() {
                    int clientSocket = socket(AF_INET, SOCK_STREAM, 0);
                    if (clientSocket == -1) {
                        cerr << "Error creating socket" << endl;
                        return;
                    }

                    sockaddr_in serverAddr;
                    serverAddr.sin_family = AF_INET;
                    serverAddr.sin_addr.s_addr = inet_addr(server_ip.c_str());
                    serverAddr.sin_port = htons(server_port);

                    if (connect(clientSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
                        cerr << "Error connecting to server" << endl;
                        close(clientSocket);
                        return;
                    }

                    auto start_time = chrono::steady_clock::now();

                    int offset = initial_offset;
                    bool eof_received = false;
                    char buffer[BUFFER_SIZE];

                    while (not eof_received && !completed.load()) {
                        string request = to_string(offset) + "\n";
                        send(clientSocket, request.c_str(), request.length(), 0);

                        int bytesRead = recv(clientSocket, buffer, BUFFER_SIZE - 1, 0);
                        if (bytesRead <= 0) {
                            cerr << "Server disconnected\n";
                            break;
                        }
                        buffer[bytesRead] = '\0';

                        string response(buffer);
                        if (response == "$$\n") {
                            break;
                        }

                        istringstream iss(response);
                        string line;
                        while (getline(iss, line, '\n')) {
                            istringstream il(line);
                            string word;
                            while (getline(il, word, ',')) {
                                if (word == "EOF") {
                                    eof_received = true;
                                    completed.store(true);
                                    cout << "received EOF\n";
                                    break;
                                }
                                if (!word.empty() && !completed.load()) {
                                    clients[index].word_count[word]++;
                                    offset++;
                                }
                            }
                            if (eof_received) break;
                        }
                    }

                    close(clientSocket);

                    auto end_time = chrono::steady_clock::now();
                    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);

                    lock_guard<mutex> lock(completion_times_mutex);
                    completion_times.push_back(duration);
                });
            }
            
            for (auto& t : threads) {
                t.join();
            }
        }
        
    }

    int getStats() {
        if (completion_times.empty()) {
            cerr << "No requests completed" << endl;
            return -1;
        }

        auto min_time = *min_element(completion_times.begin(), completion_times.end());
        auto max_time = *max_element(completion_times.begin(), completion_times.end());
        auto avg_time = accumulate(completion_times.begin(), completion_times.end(), chrono::microseconds(0)) / completion_times.size();

        cout << "Avg. Completion times: " << avg_time.count() << endl;
        int ans = avg_time.count();
        return ans;
    }

private: // Member variables
    string server_ip;
    int server_port, num_clients;
    vector<Clients> clients;
    vector<chrono::microseconds> completion_times;
    mutex completion_times_mutex;
    bool fair;
};

int main(int argc, char *argv[]) {
    json config = read_config("config.json");

    if (argc == 1) {
        int avg_time = 0;
        Client client(config);
        client.run(avg_time);
        ofstream file("avg_time.csv");
        file << avg_time;
        
    }
    if (argc > 1) {
        if (atoi(argv[1]) == 1) {
            // make run
            int avg = 0;
            ofstream file("avg_time.csv");
            Client client1(config);
            client1.run(avg);
            file << avg << ",";

            Client client2(config);
            client2.run(avg);
            file << avg;
            
        } else {
            // make fairness
            ofstream file1("avg_time.csv");
            ofstream file2("fairness_results.csv");
            int avg{};
            double jain_index{};
            Client client1(config, true);
            client1.run_fair(avg, jain_index);
            file1 << avg << ',';
            file2 << jain_index << endl;

            Client client2(config, true);
            client2.run_fair(avg, jain_index);
            file1 << avg;
            file2 << jain_index << endl;
        }
    }   
    
    return 0;
}


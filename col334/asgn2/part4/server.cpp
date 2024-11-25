#include "helper.hpp"

class Scheduler {
public:
    virtual void addRequest(const ClientRequest& request) = 0;
    virtual ClientRequest getNextRequest() = 0;
    virtual bool hasRequests() const = 0;
};

class FIFOScheduler : public Scheduler {
    void addRequest(const ClientRequest& request) override {
        lock_guard<mutex> lock(queueMutex);
        requestQueue.push(request);
    }

    ClientRequest getNextRequest() override {
        lock_guard<mutex> lock(queueMutex);
        ClientRequest request = requestQueue.front();
        requestQueue.pop();
        return request;
    }

    bool hasRequests() const override {
        return !requestQueue.empty();
    }
private:
    queue<ClientRequest> requestQueue;
    mutex queueMutex;
};

class RoundRobinScheduler : public Scheduler {
public:
    void addRequest(const ClientRequest& request) override {
        lock_guard<mutex> lock(schedulerMutex);
        if (clientQueues[request.clientSocket].empty()) {
            clientOrder.push(request.clientSocket);
        }
        clientQueues[request.clientSocket].push(request);
    }

    ClientRequest getNextRequest() override {
        lock_guard<mutex> lock(schedulerMutex);
        int clientSocket = clientOrder.front();
        clientOrder.pop();

        ClientRequest request = clientQueues[clientSocket].front();
        clientQueues[clientSocket].pop();

        if (!clientQueues[clientSocket].empty()) {
            clientOrder.push(clientSocket);
        }

        return request;
    }

    bool hasRequests() const override {
        return !clientOrder.empty();
    }
private:
    map<int, queue<ClientRequest>> clientQueues;
    queue<int> clientOrder;
    mutex schedulerMutex;
};

// the goddamm server
class Server {
public:
    Server(const json& config, unique_ptr<Scheduler> &sched)
        : server_ip(config["server_ip"]),
          server_port(config["server_port"]),
          words(load_words(config["input_file"])),
          k(config["k"]),
          p(config["p"]),
          scheduler(sched.get()),
          running(true) {}

    void start() {
        int serverSocket = socket(AF_INET, SOCK_STREAM, 0);
        if (serverSocket == -1) {
            cerr << "Error creating socket" << endl;
            return;
        }

        int opt = 1;
        if (setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
            cerr << "Error setting SO_REUSEADDR option: " << strerror(errno) << endl;
            return;
        }

        sockaddr_in serverAddr;
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_addr.s_addr = inet_addr(server_ip.c_str());
        serverAddr.sin_port = htons(server_port);

        if (bind(serverSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
            cerr << "Error binding socket: " << strerror(errno) << endl;
            close(serverSocket);
            return;
        }

        if (listen(serverSocket, SOMAXCONN) < 0) {
            cerr << "Error listening on socket" << endl;
            return;
        }

        thread schedulerThread(&Server::schedulerLoop, this);
        
        while (running) {
            sockaddr_in clientAddr;
            socklen_t clientAddrLen = sizeof(clientAddr);
            int clientSocket = accept(serverSocket, (struct sockaddr*)&clientAddr, &clientAddrLen);
            
            if (clientSocket < 0) {
                cerr << "Error accepting connection" << endl;
                continue;
            }

            thread(&Server::handleClient, this, clientSocket).detach();
        }

        schedulerThread.join();
        close(serverSocket);
    }

private:
    void handleClient(int clientSocket) {
        char buffer[BUFFER_SIZE];
        while (true) {
            int bytesRead = recv(clientSocket, buffer, BUFFER_SIZE - 1, 0);
            if (bytesRead <= 0) {
                break;
            }
            buffer[bytesRead] = '\0';

            int offset;
            sscanf(buffer, "%d", &offset);

            ClientRequest request{clientSocket, offset, chrono::steady_clock::now()};
            scheduler->addRequest(request);
            cv.notify_one();
        }
        close(clientSocket);
    }

    void schedulerLoop() {
        while (running) {
            unique_lock<mutex> lock(serverMutex);
            cv.wait(lock, [this] { return scheduler->hasRequests() || !running; });

            if (!running) break;

            ClientRequest request = scheduler->getNextRequest();
            processRequest(request);
            lock.unlock();
        }
    }

    void processRequest(const ClientRequest& request) {
        int offset = request.offset;
        int clientSocket = request.clientSocket;

        if (offset >= words.size()) {
            send(clientSocket, "$$\n", 3, 0);
            return;
        }
        int words_to_send = min(k, static_cast<int>(words.size()) - offset);
        bool last = (words.size() - offset) <= k;

        string packets = "";
        for (int i = 0; i < words_to_send; i += p) {
            string packet = "";
            int words_in_packet = min(p, words_to_send - i);

            int j = 0;
            for (; j < words_in_packet; j++) {
                packet += words[offset + i + j];
                if (j < words_in_packet - 1)
                    packet += ',';
            }
            if (last && (i + j >= words.size()))
                packet += ",EOF";
            packet += '\n';

            packets += packet;
        } 
        
        send(clientSocket, packets.c_str(), packets.length(), 0);
    }

private:
    string server_ip;
    int server_port;
    int k, p;
    vector<string> words;
    unique_ptr<Scheduler> scheduler;
    mutex serverMutex;
    condition_variable cv;
    bool running;
};

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "did not add arguement\n";
        return 1;
    }

    json config = read_config("config.json");
    int schedulerType = atoi(argv[1]);

    try {
        switch (schedulerType) {
            case 1: {
                unique_ptr<Scheduler> scheduler = make_unique<FIFOScheduler>();
                Server server(config, scheduler);
                server.start();
                break;
            }
            case 2: {
                unique_ptr<Scheduler> scheduler = make_unique<RoundRobinScheduler>();
                Server server(config, scheduler);
                server.start();
                break;
            }
            case 3: {
                unique_ptr<Scheduler> fifoScheduler = make_unique<FIFOScheduler>();
                Server fifoServer(config, fifoScheduler);
                fifoServer.start();
                
                unique_ptr<Scheduler> rrScheduler = make_unique<RoundRobinScheduler>();
                Server rrServer(config, rrScheduler);
                rrServer.start();
                break;
            }
            case 4: {
                unique_ptr<Scheduler> fifoScheduler = make_unique<FIFOScheduler>();
                Server fifoServer(config, fifoScheduler);
                fifoServer.start();
                
                unique_ptr<Scheduler> rrScheduler = make_unique<RoundRobinScheduler>();
                Server rrServer(config, rrScheduler);
                rrServer.start();
                break;
            }
            default:
                throw invalid_argument("Invalid scheduler type");
        }
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }    

    return 0;
}


import matplotlib.pyplot as plt
import json
import subprocess
import time
import numpy as np

def run_experiment(num_clients):
    with open('config.json', 'r+') as f:
        config = json.load(f)
        config['num_clients'] = num_clients
        f.seek(0)
        json.dump(config, f, indent=4)
        f.truncate()

    try:
        subprocess.run(['make', 'run'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running Makefile: {e}")

    with open('avg_time.txt', 'r') as f:
        completion_time = float(f.read())

    return completion_time

client_trials = [1, 4, 8, 12, 16, 20, 24, 28, 32]
completion_times = []

for num_clients in client_trials:
    times = run_experiment(num_clients)
    completion_times.append((times))
    print(f"{num_clients} clients done")

plt.errorbar(client_trials, completion_times, fmt='o-', capsize=5)
plt.xlabel('Number of Concurrent Clients')
plt.ylabel('Completion Time (s)')
plt.title('Completion Time vs Number of Concurrent Clients')
plt.savefig('plot.png')
plt.close()

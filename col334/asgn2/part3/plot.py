import matplotlib.pyplot as plt
import json
import subprocess
import time
import numpy as np

# Define the different protocols to be tested
protocols = ['aloha', 'beb', 'cscd']  # These correspond to the makefile targets
n_values = [1, 2, 3, 4, 5]  # Different values of n to test
completion_times = {protocol: [] for protocol in protocols}  # Store times for each protocol

def run_experiment(protocol, num_clients):
    with open('config.json', 'r+') as f:
        config = json.load(f)
        config['num_clients'] = num_clients
        f.seek(0)
        json.dump(config, f, indent=4)
        f.truncate()
    try:
        subprocess.run(['make', f'run-{protocol}'], check=True)  # Use the appropriate makefile target
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {protocol}: {e}")

    with open('avg_time.txt', 'r') as f:
        completion_time = float(f.read())
    
    return completion_time

# Run experiments for each protocol and n value
for protocol in protocols:
    for n in n_values:
        times = run_experiment(protocol, n)
        avg_time_per_client = times
        completion_times[protocol].append(avg_time_per_client)
        print(f"{protocol} with {n} clients done.")
        time.sleep(5)  # Sleep for 1 second to allow the server to restart

# Plotting results
for protocol in protocols:
    plt.plot(n_values, completion_times[protocol], marker='o', label=protocol)

plt.xlabel('Number of Clients (n)')
plt.ylabel('Avg Completion Time per Client (s)')
plt.title('Avg Completion Time per Client vs Number of Clients for Different Protocols')
plt.legend()

plt.savefig('part_3_2.png')
plt.close()

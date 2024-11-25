import matplotlib.pyplot as plt
import subprocess
import json
import csv

def run_experiment(n, policy):
    with open('config.json', 'r+') as f:
        config = json.load(f)
        config['num_clients'] = n
        f.seek(0)
        json.dump(config, f, indent=4)
        f.truncate()

    try:
        subprocess.run(['make', policy], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running Makefile: {e}")

def read_csv():
    with open('avg_time.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            avg_time = float(row[0])
            return avg_time

n_values = [1, 2, 4, 8, 16, 32]
fifo_avg_time = []
rr_avg_time = []

for n in n_values:
    times = 0
    for i in range(10):
        run_experiment(n, 'run-fifo')
        times+=read_csv()
    fifo_avg_time.append(times/10)

    times = 0
    for i in range(10):
        run_experiment(n, 'run-rr')
        times+=read_csv()
    rr_avg_time.append(times/10)

# plot    
plt.figure(figsize=(16, 10))
plt.plot(n_values, fifo_avg_time, label='FIFO', marker='o', color='blue')
plt.plot(n_values, rr_avg_time, label='Round Robin', marker='o', color='red')

plt.title('Average Completion Time vs Number of Clients')
plt.xlabel('Number of Clients (n)')
plt.ylabel('Average Completion Time (micro-seconds)')

plt.legend()
plt.grid(True)
plt.savefig('plot.png')


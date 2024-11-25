import matplotlib.pyplot as plt
import json
import subprocess
import time
import numpy as np

def run_experiment(p):
    with open('config.json', 'r+') as f:
        config = json.load(f)
        config['p'] = p
        f.seek(0)
        json.dump(config, f, indent=4)
        f.truncate()

    start_time = time.time()
    try:
        subprocess.run(['make', 'run'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running Makefile: {e}")
    end_time = time.time()
    return end_time - start_time

p_values = range(1, 11)
completion_times = []

for p in p_values:
    times = [run_experiment(p) for _ in range(10)]
    avg_time = np.mean(times)
    completion_times.append(avg_time)
    print(f"{p} done")

plt.errorbar(p_values, completion_times, yerr=std_devs, fmt='o-', capsize=5)
plt.xlabel('p (words per packet)')
plt.ylabel('Completion Time (s)')
plt.title('Completion Time vs Words per Packet')
plt.savefig('plot.png')
plt.close()

import subprocess

cmd = ['g++', '-fopenmp', '-std=c++17', 'check.cpp', './template.cpp', '-o', 'check']
subprocess.run(cmd)

powers = [6, 8, 10, 12]

times = 5

for i in powers:
    power = 2**i
    file = f'data/data_{i}.txt'
    cmd = ['./check.exe', str(power)]
    with open(file, 'a') as f:
        for _ in range(times):
            result = subprocess.run(cmd, stdout=subprocess.PIPE)
            f.write(result.stdout.decode('utf-8'))
            
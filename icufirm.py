import time
import torch
import numpy as np
from tqdm import tqdm
import pickle

device = torch.device('cuda:1')

ns = []
time_cpu = []
time_gpu = []

for i in tqdm(range(1, 1000)):
    ns.append(i)
    matrix = np.random.randn(i, i)

    start_time = time.time()
    np.linalg.inv(matrix)
    elapsed_time = time.time() - start_time
    time_cpu.append(elapsed_time)

    start_time = time.time()
    torch.linalg.inv(torch.Tensor(matrix).to(device))
    elapsed_time = time.time() - start_time
    time_gpu.append(elapsed_time)

dict = {'dim': ns, 'cpu': time_cpu, 'gpu': time_gpu}

with open('inv_exp.pickle', 'wb') as file:
    pickle.dump(dict, file, protocol=pickle.HIGHEST_PROTOCOL)

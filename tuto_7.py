#

# EVERYTHING AS TUTO6 - BEGINNING

import torch
from IPython import embed
import os
import cv2
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# EXCEPT THE FOLLOWING
from datetime import datetime
import time

training_data = np.load('training_data.npy', allow_pickle=True)

print('Cuda availability:', torch.cuda.is_available())
print('# GPUs: {}'.format(torch.cuda.device_count()))

if torch.cuda.is_available():
    device = torch.device('cuda:1')
    print('Running on GPU')
else:
    device = torch.device('cpu')
    print('Running on CPU')


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)                                        # WHY 1? 5 is the kernel size ---> kernel is 5x5
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self._to_linear = None
        x = torch.randn(50, 50).view(-1, 1, 50, 50)                             # fake input rescaled as torch wants
        self.convs(x)                                                           # Setting _to_linear variable
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        if self._to_linear is None:
            self._to_linear = np.prod(x[0].shape)#x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x                                                                # Returning x as performing the convolutional steps

    def forward(self, x):
        x = self.convs(x)                                                       # Already exist a function that compute the convolutional part
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

# Normalize data and extract test set
images = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
images = images/255                                                             # Normalization
labels = torch.Tensor([i[1] for i in training_data])
test_frac = 0.1
test_size = int(len(training_data)*test_frac)

train_x = images[:-test_size]
train_y = labels[:-test_size]
test_x = images[-test_size:]
test_y = labels[-test_size:]


net = Net().to(device)

optimizer = optim.Adam(net.parameters(), lr=1E-3)                               # Adam optimizer
loss_function = nn.MSELoss()                                                    # MSE loss function

# EVERYTHING AS TUTO6 - END

def fwd_pass(x, y, train=False):
    if train:
        optimizer.zero_grad()
    outputs = net(x)
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True)/len(matches)
    loss = loss_function(outputs, y)
    if train:
        loss.backward()
        optimizer.step()
    return acc, loss

def test(size=32):
    random_start = np.random.randint(len(test_x)-size)
    x, y = test_x[random_start:random_start+size], test_y[random_start:random_start+size]
    with torch.no_grad():
        acc_val, loss_val = fwd_pass(x.view(-1, 1, 50, 50).to(device), y.to(device))
    return acc_val, loss_val

curr_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

model_name = 'Model_' + curr_datetime
print(model_name)

def train():
    batch_size = 100
    epochs = 30
    with open('model30.log', 'a') as file:
        for epoch in range(epochs):
            for i in tqdm(range(0, len(train_x), batch_size)):
                batch_x = train_x[i:i+batch_size].view(-1, 1, 50, 50).to(device)
                batch_y = train_y[i:i+batch_size].to(device)
                acc, loss = fwd_pass(batch_x, batch_y, train=True)
                #if i%5*batch_size == 0:
                acc_val, loss_val = test(size=100)
                file.write(model_name + ',{},{},{},{},{}\n'.format(round(time.time(), 3), round(acc, 3), round(float(loss), 5), round(acc_val, 3), round(float(loss_val), 5)))

train()

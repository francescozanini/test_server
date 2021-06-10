import torch
from IPython import embed
import os
import cv2
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

training_data = np.load('training_data.npy', allow_pickle=True)

print('Cuda availability:', torch.cuda.is_available())
print('# GPUs: {}'.format(torch.cuda.device_count()))

if torch.cuda.is_available():
    device = torch.device('cuda:1')
    print('Running on GPU')
else:
    device = torch.device('cpu')
    print('Running on CPU')


# EVERYTHING AS TUTO5 - BEGINNING

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

batch_size = 100
epochs = 10

# EVERYTHING AS TUTO5 - END


net = Net().to(device)

optimizer = optim.Adam(net.parameters(), lr=1E-3)                               # Adam optimizer
loss_function = nn.MSELoss()                                                    # MSE loss function

def train(net):
    for epoch in range(epochs):
        for i in tqdm(range(0, len(train_x), batch_size)):
            batch_x = train_x[i:i+batch_size].view(-1, 1, 50, 50)                   # why there is this 1?
            batch_y = train_y[i:i+batch_size]

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            output = net(batch_x)
            loss = loss_function(output, batch_y)
            loss.backward()
            optimizer.step()

        print('Epoch {}, loss: {}'.format(epoch, loss))

def test(net):
    correct = 0; total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_x))):
            real_class = torch.argmax(test_y[i]).to(device)
            net_out = net(test_x[i].view(-1, 1, 50, 50).to(device)).squeeze()
            predicted_class = torch.argmax(net_out)
            if predicted_class == real_class:
                correct += 1
            total += 1
    print('Accuracy: {}'.format(round(correct/total, 3)))

train(net)
test(net)

embed()

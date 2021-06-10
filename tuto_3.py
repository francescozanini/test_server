# Actual training of NN


# SAME AS TUTO2 - BEGINNING

import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

# Again train and test
train = datasets.MNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST('', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

# Again preprocess and load data
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

# NN class for MNIST dataset with default initialization
class Net(nn.Module):

    def __init__(self):
        h1 = 64; h2 = 64; h3 = 64; classes = 10
        super().__init__()
        self.fc1 = nn.Linear(28*28, h1)                                         # input is vectorized dimension of image
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, classes)

    def forward(self, x):                                                       # using REctified Linear Units as neurons
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)                                                         # output does not have activation
        return F.log_softmax(x, dim=1)                                          # softmax distribution (dim=1 to keep it a distribution)


net = Net()

# SAME AS TUTO2 - END


import torch.optim as optim
# First argmuent is anything the optimizer can change, and net.parameters() is everything that can be changed inside tne net, so this optimizer is allowed to change everything
optimizer = optim.Adam(net.parameters(), lr=1E-3)
# Epochs are how many times we RETRAIN the net on the SAME inputs.
epochs = 3

# Training the net
for epoch in range(epochs):
    for data in trainset:
        Images, labels = data
        net.zero_grad()                                                         # Set the gradient to zero - Stop summing up things (implement the findings)
        output = net(Images.view(-1, 28*28))
        loss = F.nll_loss(output, labels)                                       # Compute negative log-likelihood loss on labels of batch
        loss.backward()                                                         # Propagates gradient through net
        optimizer.step()                                                        # Actually changes the weights and "train" the net
    print(loss)                                                                 # To see of at least decreases over time

# Check accuracy on training set
correct = 0; total = 0
with torch.no_grad():                                                           # in order not to compute gradients
    for data in trainset:
        Images, labels = data
        output = net(Images.view(-1, 28*28))
        for idx, i in enumerate(output):
            if torch.argmax(i) == labels[idx]:
                correct += 1
            total += 1
print('Accuracy on training set: {}'.format(round(correct/total, 3)))
'''
# Visual check
import matplotlib.pyplot as plt
plt.imshow(Images[0].view(28, 28))
plt.show()
print(torch.argmax(net(Images[0].view(-1, 28*28))[0]))
'''
# Check accuracy on test set
correct = 0; total = 0
with torch.no_grad():                                                           # in order not to compute gradients
    for data in testset:
        Images, labels = data
        output = net(Images.view(-1, 28*28))
        for idx, i in enumerate(output):
            if torch.argmax(i) == labels[idx]:
                correct += 1
            total += 1
print('Accuracy on test set: {}'.format(round(correct/total, 3)))

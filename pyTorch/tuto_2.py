# Building first NN, no training, no loss

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
print(net)

# Faking input to pass through the net
Image = torch.rand((28, 28))
Image = Image.view(-1, 28*28)
# Always needed to reshape the input like that, 1 could be just fine but -1 should always work and stands for 'anything'
output = net(Image)                                                             # forward method is automatically called I think
print(output)

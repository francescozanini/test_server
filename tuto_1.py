# basics of data for pyTorch, usage of MNIST dataset, still no NN

from IPython import embed
import torch
import torchvision
from torchvision import transforms, datasets


# Train and Test from dataset; transforms are data-preprocessors
train = datasets.MNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST('', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

# Actual data: need to load  and pre-process data into pyTorch with possibly a batch size and shuffling them
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

# Check first batch (due to break)
for data in trainset:
    print(data)
    break

# Visualize data: first image in first batch
import matplotlib.pyplot as plt
print('This should be a {}!'.format(data[1][0]))
_, shape1, shape2 = data[0][0].shape
plt.imshow(data[0][0].view(shape1, shape2))
# This is due to the fact that pyTorch has 3D inputs with size (1, 28, 28), so we want to remove the 1 to see the image
plt.show()

# Is the dataset balanced? (Similar number of different labels)
total = 0
dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

for data in trainset:
    images, labels = data                                                       # Images and labels corresponding to the current batch
    for y in labels:                                                            # there will be batch_size of them
        dict[int(y)] += 1                                                       # Casting to int because it is actually a Tensor
        total += 1

print(dict)

# To see percentages
for i in dict:
    print('{}: {}'.format(i, dict[i]/total*100))

embed()

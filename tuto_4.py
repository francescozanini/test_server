# Data preprocessing on Microsoft Cats VS Dogs dataset

import os
import cv2
import numpy as np
from tqdm import tqdm
from IPython import embed


rebuild_data = False

class DogsVSCats():
    img_size = 50
    cats = os.path.join('PetImages', 'Cat')
    dogs = os.path.join('PetImages', 'Dog')
    labels = {cats: 0, dogs: 1}
    training_data = []
    catcount = 0
    dogcount = 0

    def make_training_data(self):
        for label in self.labels:
            print(label)
            for file in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, file)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.img_size, self.img_size))
                    self.training_data.append([np.array(img), np.eye(2)[self.labels[label]]])
                    embed()
                    if label == self.cats:
                        self.catcount += 1
                    elif label == self.dogs:
                        self.dogcount += 1
                except Exception as e:
                    #print('An error has occured, that was expected. Here is the error:')
                    #print(e)
                    pass

        np.random.shuffle(self.training_data)
        np.save('training_data.npy', self.training_data)
        print('Cats: {}\nDogs: {}'.format(self.catcount, self.dogcount))


if rebuild_data:
    dogsVScats = DogsVSCats()
    dogsVScats.make_training_data()

training_data = np.load('training_data.npy', allow_pickle=True)
print(len(training_data))

# Check labels
import matplotlib.pyplot as plt
index = 2
print(training_data[index][1])
plt.imshow(training_data[index][0], cmap='gray')
plt.show()

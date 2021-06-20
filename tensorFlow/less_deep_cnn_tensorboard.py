# Training visualisation with tensorBoard


# SAME AS TUTO2 - BEGINNING

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Activation, Conv2D
import pickle
from IPython import embed
import numpy as np


# Load preprocessed data
with open('training_data_x.pickle', 'rb') as file:
    x = pickle.load(file)
with open('training_data_y.pickle', 'rb') as file:
    y = pickle.load(file); y = np.array(y)                                      # necessary to deal with arrays to use the validation_split function

# Check on normalize function
x = x/255
another_x = x*255/267.3106806695161
x_keras = tf.keras.utils.normalize(x, axis=1)
print('Does _normalize_ function do what I have been told it does?')
if (another_x == x_keras).all():
    print('Yes!')
else:
    print('Absolutely NOT!')
print('Well, it seems that it divides for a different factor different numbers (ex. 267.3106806695161≠255)... Strange...')

# Building the NN
model = Sequential()
# First layer
model.add(Conv2D(64, (3, 3), input_shape=x.shape[1:]))                          # Convolutional layer with 64 neurons and kernel size = 3x3, [1:] is because we are neglecting the #images from the shape we pass, as it is not necessary for the NN
model.add(Activation('relu'))                                                   # Activation functions can be defined as layers !!!
model.add(MaxPooling2D(pool_size=(2, 2)))                                       # Pooling domain of size 2x2
# Second layer
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Third layer
model.add(Flatten())                                                            # To use a dense layer (i.e. fully connected)
'''
# No dense layer here!
model.add(Dense(64))
model.add(Activation('relu'))
'''
# WHY NO ACTIVATION HERE ??? - AS EXPECTED IT SHOULD BE THERE :)
# Output layer
model.add(Dense(1))                                                             # We predict directly the class through sigmoid output
model.add(Activation('sigmoid'))

# Optimizer, loss, metrics
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x, y, batch_size=32, validation_split=0.1, epochs=3)                  # batch_size are # of samples before actual update, validation_split is # of samples left for test
print('Automatically gives accuracy and loss on test set!')

# SAME AS TUTO2 - END


from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
import os

name = 'cats-vs-dogs-cnn-64_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
tensorboard = TensorBoard(log_dir=os.path.join('logs', name))

# Train the model specifying tensorboard callback
model.fit(x, y, batch_size=32, validation_split=0.1, epochs=20, callbacks=[tensorboard])

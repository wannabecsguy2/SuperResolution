import os, cv2, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

ROWS = 64
COLS = 64
CHANNELS = 3
TRAIN_DIR = 'training_set/'
TEST_DIR = 'test_set/'

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
test_images = [TEST_DIR+i for i in os.listdir(TEST_DIR)]

#Function to read and reshape images
def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    return cv2.resize(img, (ROWS, COLS), interpolation = cv2.INTER_CUBIC)

#Function to prepare dataset
def prep_data(images):
    m = len(images)
    n_x = ROWS*COLS*CHANNELS
    
    X = np.ndarray((m, ROWS, COLS, CHANNELS), dtype = np.uint8)
    y = np.zeros((m, 1))
    print('X.shape is {}'.format(X.shape))

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        X[i,:] = np.squeeze(image.reshape((ROWS, COLS, CHANNELS)))
        if 'dog' in image_file.lower():
            y[i, 0] = 1
        elif 'cat' in image_file.lower():
            y[i, 0] = 0

        if i%1000 == 0:
            print('Proceed of {} of {}'.format(i, m))
    
    return X, y

#Function to show an image in the dataset given its index
def show_image(X, y, idx) :
  image = X[idx]
  #image = image.reshape((ROWS, COLS, CHANNELS))
  plt.figure(figsize=(4,2))
  plt.imshow(image)
  plt.title("This is a {}".format(classes[y[idx,0]]))
  plt.show()

X_train, y_train = prep_data(train_images)
X_test, y_test = prep_data(test_images)

classes = {1 : 'Dog', 0 : 'Cat'}

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 75)

y_train_one_hot = to_categorical(y_train)

y_val_one_hot = to_categorical(y_val)

show_image(X_train, y_train, 0)

X_train_norm = X_train/255
X_val_norm = X_val/255

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = (ROWS, COLS, CHANNELS), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, (1, 1), activation = 'relu'))

model.add(Flatten())
model.add(Dropout(0.4))

model.add(Dense(units = 120, activation = 'relu'))
model.add(Dense(units = 2, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()

model.fit(X_train_norm, y_train_one_hot, validation_data = (X_val_norm, y_val_one_hot), epochs = 50, batch_size = 64)

image = X_test[0]
test_pred = model.predict(image.reshape(1, 64, 64, 3))
print(test_pred)
show_image(X_test, y_test, 0)
print("Our Model Prediction: {}".format(test_pred))
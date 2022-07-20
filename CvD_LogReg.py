import os, cv2, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV

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
    
    X = np.ndarray((n_x, m), dtype = np.uint8)
    y = np.zeros((1, m))
    print('X.shape is {}'.format(X.shape))

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        X[:,i] = np.squeeze(image.reshape((n_x, 1)))
        if 'dog' in image_file.lower():
            y[0, i] = 1
        elif 'cat' in image_file.lower():
            y[0, i] = 0

        if i%1000 == 0:
            print('Proceed of {} of {}'.format(i, m))
    
    return X, y

#Function to show an image in the dataset given its index
def show_image(X, y, idx):
    X_use, y_use = X.T, y.T
    image = X_use[idx]
    image.reshape((ROWS, COLS, CHANNELS))
    plt.figure(figsize = (4, 2))
    plt.imshow(image)
    plt.title('This is a {}'.format(classes[y_use[idx, 0]]))
    plt.show()

X_train, y_train = prep_data(train_images)
X_test, y_test = prep_data(test_images)

classes = {1 : 'Dog', 0 : 'Cat'}

clf = LogisticRegressionCV(solver = 'liblinear')

X_train_lr, y_train_lr = X_train.T, y_train.T.ravel()
X_test_lr, y_test_lr = X_test.T, y_test.T.ravel()
clf.fit(X_train_lr, y_train_lr)

print('Model Accuracy: {:2f}%'.format(clf.score(X_test_lr, y_test_lr)*100))

#Function to predict image class
def show_image_prediction(image, model = clf):
    image = image.reshape(1, -1)
    image_class = classes[model.predict(image).item()]
    image = image.reshape((ROWS, COLS, CHANNELS))
    plt.figure(figsize = (4,2))
    plt.imshow(image)
    plt.title("Test : I think this is {}".format(image_class))
    plt.show()

test_image = X_test_lr[0]
show_image_prediction(test_image)
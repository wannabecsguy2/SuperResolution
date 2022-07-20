import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from sklearn.decomposition import PCA
digits = load_digits()
plt.figure(figsize=(20,4))
"""
Plotting:

for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize = 20)
    plt.show()
"""
LogReg = LogisticRegression(solver = 'liblinear', max_iter = 500)
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.2, random_state = 82)

LogReg.fit(X_train, y_train)
print(X_test[0].reshape(1, -1).shape)
print(X_test[0].shape)
print(y_test[0] - LogReg.predict(X_test[0].reshape(1, -1)))

def predict(X_data):
    
    X_data = X_data.reshape(X_data.shape[0], 784)
    #PCA Transform for Dimensionality Reduction
    transform = PCA(n_components = 64).fit_transform(X_data)
    predictions = []
    for i in range(len(transform)):
        predictions.append(LogReg.predict(transform[i].reshape(1, -1)))
    return predictions

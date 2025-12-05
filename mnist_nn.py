import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['original', 'fashion'])
arguments = parser.parse_args()

# Load MNIST or Fashion MNIST dataset based on cmd line argument
if arguments.dataset == 'original': 
    mnist = tf.keras.datasets.mnist
else:
    mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape images
x_train_reshape = x_train.reshape(x_train.shape[0], 28 * 28)
x_test_reshape = x_test.reshape(x_test.shape[0], 28 * 28)

# Normalize pixel values
x_train_reshape = x_train_reshape / 255.0
x_test_reshape = x_test_reshape / 255.0


knn_classifier = KNeighborsClassifier(n_neighbors=1)
knn_classifier.fit(x_train_reshape, y_train)
y_prediction = knn_classifier.predict(x_test_reshape)

#evaluation function
def calculate_accuracy(pred, gt):
    correct_pred = np.sum(pred == gt)
    accuracy = (correct_pred / len(gt)) * 100
    return accuracy

accuracy = class_acc(y_prediction, y_test)
print(f"Classification accuracy is {accuracy:.2f}%")



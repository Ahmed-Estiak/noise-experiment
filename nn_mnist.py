import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data_type', choices=['original', 'fashion'])
args = parser.parse_args()

# Load MNIST or Fashion MNIST dataset based on command line input
dataset = tf.keras.datasets.mnist if args.data_type == 'original' else tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = dataset.load_data()

# Flatten images
train_flat = train_images.reshape(train_images.shape[0], -1)
test_flat = test_images.reshape(test_images.shape[0], -1)


knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(train_flat, train_labels)
predictions = knn_model.predict(test_flat)

def class_acc(pred, gt):
    correct = np.sum(pred == gt )
    return (correct / len(gt)) * 100 

accuracy_score = class_acc(predictions, test_labels)
print(f"Classification accuracy is {accuracy_score:.2f}%")
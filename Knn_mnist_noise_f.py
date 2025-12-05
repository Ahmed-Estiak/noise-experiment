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

# Normalize to [0, 1] as float32
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

# Add Gaussian noise to training data only
np.random.seed(0)
noise = np.random.normal(loc=0.0, scale=0.05, size=train_images.shape)
train_images_noisy = np.clip(train_images + noise, 0.0, 1.0)

# Flatten images
train_flat = train_images_noisy.reshape(train_images.shape[0], -1)
test_flat = test_images.reshape(test_images.shape[0], -1)


knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(train_flat, train_labels)
predictions = knn_model.predict(test_flat)

def class_acc(pred, gt):
    correct = np.sum(pred == gt )
    return (correct / len(gt)) * 100 

accuracy_score = class_acc(predictions, test_labels)
print(f"Classification accuracy is {accuracy_score:.2f}%")

import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal
import argparse

# Parse command line argument
parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['original', 'fashion'])
args = parser.parse_args()

# Load dataset based on command line input
if args.dataset == 'original':
    mnist = tf.keras.datasets.mnist
else:
    mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the training and test sets
x_train = x_train.reshape(x_train.shape[0], 28*28).astype(np.float32)
x_test = x_test.reshape(x_test.shape[0], 28*28).astype(np.float32)

# Normalize the pixel values
x_train /= 255.0
x_test /= 255.0

# Function to compute classification accuracy
def class_acc(pred, gt):
    pred = np.array(pred)
    gt = np.array(gt)
    correct_predictions = np.sum(pred == gt)
    accuracy = (correct_predictions / len(gt)) * 100
    return accuracy

# Function to calculate log likelihoods and predictions
def calculate_predictions(x_train_noise, x_test, y_train):
    number_classes = 10
    number_features = x_train_noise.shape[1]

    means = np.zeros((number_classes, number_features))
    covariances = np.zeros((number_classes, number_features, number_features))

    # Compute mean and covariance for each class
    for k in range(number_classes):
        class_k_samples = x_train_noise[y_train == k]
        means[k, :] = np.mean(class_k_samples, axis=0)
        covariances[k, :, :] = np.cov(class_k_samples.T)

    log_likelihoods = np.zeros((x_test.shape[0], number_classes))

    for k in range(number_classes):
        log_likelihoods[:, k] = multivariate_normal.logpdf(x_test, mean=means[k], cov=covariances[k])

    predicted_classes = np.argmax(log_likelihoods, axis=1)
    return predicted_classes

# If Fashion MNIST is selected, iterate over different noise scales
if args.dataset == 'fashion':

    # Add Gaussian noise to the training data
    np.random.seed(0)
    noise1 = np.random.normal(loc=0.0, scale=0.15, size=x_train.shape)
    x_train_noise = x_train + noise1

    noise2 = np.random.normal(loc=0.0, scale=0.14, size=x_test.shape)

    x_test_noise=x_test + noise2

    # Get predicted classes for this noise level
    predicted_classes = calculate_predictions(x_train_noise, x_test_noise, y_train)

    # Calculate and print accuracy
    accuracy = class_acc(predicted_classes, y_test)
    print(f'Accuracy: {accuracy:.2f}%')

else:
    noise3 = np.random.normal(loc=0.0, scale=0.27, size=x_train.shape)
    x_train_noise = x_train + noise3
    noise4 = np.random.normal(loc=0.0, scale=0.05, size=x_test.shape)

    x_test_noise=x_test + noise4
    # For original MNIST, no noise is added
    predicted_classes = calculate_predictions(x_train_noise, x_test_noise, y_train)

    # Calculate accuracy
    accuracy = class_acc(predicted_classes, y_test)
    print(f'Classification accuracy for original MNIST: {accuracy:.2f}%')


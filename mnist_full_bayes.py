import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['original', 'fashion'])
args = parser.parse_args()

if args.dataset == 'original':
    mnist = tf.keras.datasets.mnist
else:
    mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(x_train.shape[0], 28*28).astype(np.float32)
x_test = x_test.reshape(x_test.shape[0], 28*28).astype(np.float32)

# Normalize the pixel values
x_train /= 255.0
x_test /= 255.0

# Add Gaussian noise to the training data
np.random.seed(0)
noise = np.random.normal(loc=0.0, scale=0.27, size=x_train.shape)
x_train_noise = x_train + noise

number_classes = 10
number_features = x_train.shape[1]  

means = np.zeros((number_classes, number_features))
covariances = np.zeros((number_classes, number_features, number_features))


for k in range(number_classes):
    class_k_samples = x_train_noise[y_train == k]
    means[k, :] = np.mean(class_k_samples, axis=0)
    covariances[k, :, :] = np.cov(class_k_samples.T) 

log_likelihoods = np.zeros((x_test.shape[0], number_classes))



for k in range(number_classes):
    log_likelihoods[:, k] = multivariate_normal.logpdf(x_test, mean=means[k], cov=covariances[k])

predicted_classes = np.argmax(log_likelihoods, axis=1)


def class_acc(pred, gt):
    pred = np.array(pred)
    gt = np.array(gt)
    
    correct_predictions = np.sum(pred == gt)
    accuracy = (correct_predictions / len(gt)) * 100
    
    return accuracy


accuracy = class_acc(predicted_classes, y_test)
print(f'Classification accuracy is {accuracy:.2f}%')

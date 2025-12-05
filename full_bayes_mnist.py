import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['original', 'fashion'])
args = parser.parse_args()

if args.dataset == 'original':
    data_source = tf.keras.datasets.mnist
else:
    data_source = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data_source.load_data()

train_flat = train_images.reshape(train_images.shape[0], -1).astype(np.float32)
test_flat = test_images.reshape(test_images.shape[0], -1).astype(np.float32)

np.random.seed(0)
random_noise = np.random.normal(loc=0.0, scale=0.2, size=train_flat.shape)
train_with_noise = train_flat + random_noise

num_classes = 10
num_features = train_flat.shape[1]

mean_vectors = np.zeros((num_classes, num_features))
covariance_matrices = np.zeros((num_classes, num_features, num_features))

for class_index in range(num_classes):
    class_samples = train_with_noise[train_labels == class_index]
    mean_vectors[class_index, :] = np.mean(class_samples, axis=0)
    covariance_matrices[class_index, :, :] = np.cov(class_samples.T)

log_probs = np.zeros((test_flat.shape[0], num_classes))

for class_index in range(num_classes):
    log_probs[:, class_index] = multivariate_normal.logpdf(test_flat, mean=mean_vectors[class_index], cov=covariance_matrices[class_index])

predicted_classes = np.argmax(log_probs, axis=1)

def class_acc(pred, gt):
    pred = np.array(pred)
    gt = np.array(gt)
    
    correct_predictions = np.sum(pred == gt)
    accuracy = (correct_predictions / len(gt)) * 100
    
    return accuracy

accuracy_score = class_acc(predicted_classes, test_labels)
print(f'Classification accuracy is {accuracy_score:.2f}%')

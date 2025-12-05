import numpy as np
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['original', 'fashion'])
args = parser.parse_args()

if args.dataset == 'original':
    dataset = tf.keras.datasets.mnist
else:
    dataset = tf.keras.datasets.fashion_mnist

(train_data, train_labels), (test_data, test_labels) = dataset.load_data()

# Reshape the training and test sets
train_flat = train_data.reshape(train_data.shape[0], -1).astype(np.float32)
test_flat = test_data.reshape(test_data.shape[0], -1).astype(np.float32)


np.random.seed(0)
random_noise = np.random.normal(loc=0.0, scale=0.1, size=train_flat.shape)
train_with_noise = train_flat + random_noise

num_classes = 10
num_features = train_flat.shape[1]

mean_vectors = np.zeros((num_classes, num_features))
variance_vectors = np.zeros((num_classes, num_features))

for class_index in range(num_classes):
    class_samples = train_with_noise[train_labels == class_index]
    mean_vectors[class_index, :] = np.mean(class_samples, axis=0)
    variance_vectors[class_index, :] = np.var(class_samples, axis=0)

def compute_log_likelihood(sample, class_idx):
    mean = mean_vectors[class_idx]
    variance = variance_vectors[class_idx]
    log_likelihood = -0.5 * (np.sum(np.log(2 * np.pi * variance)) + np.sum(((sample - mean) ** 2) / variance))
    return log_likelihood

def class_acc(pred, gt):
    pred = np.array(pred)
    gt = np.array(gt)
    
    correct_predictions = np.sum(pred == gt)
    accuracy = (correct_predictions / len(gt)) * 100
    
    return accuracy

predictions = []
for sample in test_flat:
    log_likelihoods = np.array([compute_log_likelihood(sample, k) for k in range(num_classes)])
    predicted_class = np.argmax(log_likelihoods)
    predictions.append(predicted_class)

accuracy_score = class_acc(predictions, test_labels)
print(f'Classification accuracy is {accuracy_score:.2f}%')

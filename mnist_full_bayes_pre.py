import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal
from random import random
# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the training and test sets from (28, 28) to (784,)
x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32)
x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32)

# Normalize the pixel values to range [0, 1]
x_train /= 255.0
x_test /= 255.0

# Add Gaussian noise to the training data
noise = np.random.normal(loc=0.0, scale=0.1, size=x_train.shape)
x_train_noisy = x_train + noise

# Number of classes
num_classes = 10
num_features = x_train.shape[1]  # 784 features

# Calculate mean vectors and full covariance matrices for each class
means = np.zeros((num_classes, num_features))
covariances = np.zeros((num_classes, num_features, num_features))
print('hello1')

for k in range(num_classes):
    class_k_samples = x_train_noisy[y_train == k]
    means[k, :] = np.mean(class_k_samples, axis=0)
    covariances[k, :, :] = np.cov(class_k_samples.T) + .014 * np.eye(num_features)  # Regularize covariance matrix
    #xyz=class_k_samples + .014 * np.eye(num_features)
    #covariances[k, :, :] = np.cov(xyz.T)
print('hello2')

# Function to compute log likelihood using full multivariate normal distribution (Equation 4)
def compute_log_likelihood(x, class_idx):
    mean = means[class_idx]
    cov = covariances[class_idx]
    log_likelihood = multivariate_normal.logpdf(x, mean=mean, cov=cov)
    return log_likelihood
print('hello3')
# Classify the test samples using the highest log likelihood
correct_predictions = 0
a=0
b=0
c=0
for i in range(x_test.shape[0]):
    log_likelihoods = np.array([compute_log_likelihood(x_test[i], k) for k in range(num_classes)])
    
    a=a+1
    print(a)

    predicted_class = np.argmax(log_likelihoods)  # Class with the highest log likelihood
    
    c=c+1
    print(c)
    if predicted_class == y_test[i]:
        correct_predictions += 1

        b=b+1
        print(b)
        '''for i in range(x_test.shape[0]):
    log_likelihoods = np.array([compute_log_likelihood(x_test[i], k) for k in range(num_classes)])
    predicted_class = np.argmax(log_likelihoods)  # Class with the highest log likelihood
    if predicted_class == y_test[i]:
        correct_predictions += 1'''

print('hello5')
# Calculate accuracy
accuracy = correct_predictions / x_test.shape[0]
print(f'Classification accuracy with full multivariate Gaussian: {accuracy * 100:.2f}%')


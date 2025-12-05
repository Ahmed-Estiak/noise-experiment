import numpy as np
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['original', 'fashion'])
args = parser.parse_args()

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

np.random.seed(0)
noise = np.random.normal(loc=0.0, scale=0.14, size=x_train.shape)
x_train_noise = x_train + noise



number_classes = 10
number_features = x_train.shape[1]  


means = np.zeros((number_classes, number_features))
variances = np.zeros((number_classes, number_features))

for k in range(number_classes):
    class_k_samples = x_train_noise[y_train == k]
    means[k, :] = np.mean(class_k_samples, axis=0)
    variances[k, :] = np.var(class_k_samples, axis=0)

def compute_log_likelihood(x, class_index):
    mean = means[class_index]
    variance = variances[class_index]
    log_likelihood = -0.5 * (np.sum(np.log(2 * np.pi * variance)) + np.sum(((x - mean) ** 2) / variance))
    return log_likelihood

#evaluation function
def class_acc(pred, gt):
    pred = np.array(pred)
    gt = np.array(gt)
    
    correct_predictions = np.sum(pred == gt)
    accuracy = (correct_predictions / len(gt)) * 100
    
    return accuracy

y_pred = []
for i in range(x_test.shape[0]):
    log_likelihoods = np.array([compute_log_likelihood(x_test[i], k) for k in range(number_classes)])
    predicted_class = np.argmax(log_likelihoods) 
    y_pred.append(predicted_class)
 

accuracy = class_acc(y_pred, y_test)
print(f'Classification accuracy is {accuracy:.2f}%')

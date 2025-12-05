import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD


def one_hot_encoder(tags, num_classes=10):
    one_hot_vec = np.zeros((len(tags), num_classes))
    for index, label in enumerate(tags):
        one_hot_vec[index, label] = 1
    return one_hot_vec


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape((x_train.shape[0], 784)).astype('float32') / 255.0
x_test = x_test.reshape((x_test.shape[0], 784)).astype('float32') / 255.0

# Add Gaussian noise to training and test data
np.random.seed(0)
train_noise = np.random.normal(loc=0.0, scale=0.08, size=x_train.shape)
test_noise = np.random.normal(loc=0.0, scale=0.08, size=x_test.shape)
x_train_noisy = np.clip(x_train + train_noise)
x_test_noisy = np.clip(x_test + test_noise)

y_train_ohe = one_hot_encoder(y_train)
y_test_ohe = one_hot_encoder(y_test)

model = Sequential()
model.add(Dense(21, input_dim=784, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))

learning_rate = 0.16
model.compile(optimizer=SGD(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 20
history = model.fit(x_train_noisy, y_train_ohe, epochs=epochs, batch_size=16, verbose=0)

train_loss, train_accuracy = model.evaluate(x_train_noisy, y_train_ohe, verbose=0)
test_loss, test_accuracy = model.evaluate(x_test_noisy, y_test_ohe, verbose=0)

print(f'Accuracy for the training data: {train_accuracy * 100:.2f}%')
print(f'Accuracy for the test data: {test_accuracy * 100:.2f}%')

# Save the training loss curve instead of blocking on a GUI window
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training Loss Curve (Fashion MNIST)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_loss_fashion.png')
plt.close()
print('Training loss curve saved to training_loss_fashion.png')

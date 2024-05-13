import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# NOTE: this test case is y=x, such that on or to the left of the line is 0, otherwise 1

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, activation='relu', input_shape=(2,), # calculate difference between components
                          kernel_initializer=tf.constant_initializer([[-1.0], [1.0]]),
                          bias_initializer=tf.constant_initializer([0.0])),

    # Convert to 0 or 1
    tf.keras.layers.Dense(units=2, activation='relu', # convert difference to 2D vector with diff and diff - 1
                          kernel_initializer=tf.constant_initializer([[1.0, 1.0]]),
                          bias_initializer=tf.constant_initializer([0.0, -1.0])),
    tf.keras.layers.Dense(units=1, activation='relu',
                          kernel_initializer=tf.constant_initializer([[1.0], [-1.0]]),
                          bias_initializer=tf.constant_initializer([0.0])),
])

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# input = np.array([[0, 0]]) # (x, y)
# output = model.predict(input)
# print(output)

# tf.saved_model.save(model, '../saved_models/diagonal-split_classif_nnet')

BOUND = 10
X, Y = np.meshgrid(np.arange(-BOUND, BOUND+1), np.arange(-BOUND, BOUND+1))
inputs = np.column_stack((X.ravel(), Y.ravel()))
outputs = model.predict(inputs)

plt.figure(1)
plt.scatter(inputs[:, 0], inputs[:, 1], c=outputs, cmap='bwr')
plt.title('y=x NN output')
plt.xlabel('input_0')
plt.ylabel('input_1')
plt.colorbar(label='output')
plt.grid(True)
plt.show()
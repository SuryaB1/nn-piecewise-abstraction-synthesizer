import tensorflow as tf
from tensorflow import keras
import numpy as np

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=8, activation='relu', input_shape=(4,), # (0,0), (0,1), (1,0), (1,1)
                          kernel_initializer=tf.constant_initializer([[1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0]]),
                          bias_initializer=tf.constant_initializer([-1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0])),
    # tf.keras.layers.Dense(units=1, activation='sigmoid', # output is 1 for in unit square, 0 otherwise
    #                       kernel_initializer=tf.constant_initializer([[-1.0], [-1.0], [-1.0], [-1.0]]),
    #                       bias_initializer=tf.constant_initializer([0.0]))

    tf.keras.layers.Dense(units=1, activation='relu', # output is 0 for in unit square, 1 otherwise
                          kernel_initializer=tf.constant_initializer([[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]),
                          bias_initializer=tf.constant_initializer([0.0])),
])

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

input = np.array([[-1, 0.5, 0.5, 0]])
output = model.predict(input)
print(output)

tf.saved_model.save(model, 'data/inputs/models/saved_models/4D_unit_sqr_classif_nn')
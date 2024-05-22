import tensorflow as tf
from tensorflow import keras
import numpy as np

# TODO: Need to fix to perform union (not intersection) between two regions (see TODO.md)

# if in either convex trapezoid or concave pentagon, 1, otherwise 0
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=9, activation='relu', input_shape=(2,), # first four weights and biases from (convex) trapezoid, rest from (concave) pentagon
                          kernel_initializer=tf.constant_initializer([[0.0, 0.0, -1.0, 1.0, 0.0, 1.0, -1.0, -1.0, 1.0], 
                                                                      [1.0, -1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 1.0, 1.0]]),
                          bias_initializer=tf.constant_initializer([-2.0, 0.0, 0.0, -5.0, 0.0, 5.0, -1.0, 0.0, 6.0])),

    # Convert to 0 or 1
    tf.keras.layers.Dense(units=18, activation='relu', # convert difference to 2D vector with diff and diff - 1
                          kernel_initializer=tf.constant_initializer([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                                                                      [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                                                                      [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]]),
                          bias_initializer=tf.constant_initializer([0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0])),
    tf.keras.layers.Dense(units=9, activation='relu',
                          kernel_initializer=tf.constant_initializer([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                                                                      [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                                                                      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                                                                      [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                                                                      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                                                                      [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                                                                      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                                                                      [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                                                                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                                                      [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                                                      [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 
                                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]]),
                          bias_initializer=tf.constant_initializer([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),

    tf.keras.layers.Dense(units=2, activation='relu', # one neuron for y<0, x>-5, x<-1, y>x, other for y<0, x>-5, x<-1, y<-x-6
                          kernel_initializer=tf.constant_initializer([[1.0, 1.0], 
                                                                      [1.0, 1.0],
                                                                      [1.0, 1.0],
                                                                      [1.0, 1.0],
                                                                      [1.0, 1.0],
                                                                      [1.0, 1.0], 
                                                                      [1.0, 1.0], 
                                                                      [1.0, 0.0], 
                                                                      [0.0, 1.0]]),
                          bias_initializer=tf.constant_initializer([-7.0, -7.0])),
    tf.keras.layers.Dense(units=1, activation='relu',
                          kernel_initializer=tf.constant_initializer([[1.0], [1.0]]),
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

input = np.array([[2, 0.5]])
output = model.predict(input)
print(output)

tf.saved_model.save(model, '../saved_models/disjoint-polys_classif_nnet')
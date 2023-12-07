import tensorflow as tf
from tensorflow import keras
import numpy as np

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=3, activation='relu', input_shape=(3,),
                          kernel_initializer=tf.constant_initializer([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                          bias_initializer=tf.constant_initializer([0.0, 0.0, 0.0])),
    tf.keras.layers.Dense(units=2, activation='linear',
                          kernel_initializer=tf.constant_initializer([[3.0, 0.0], [2.0, 0.0], [0.0, 1.0]]),
                          bias_initializer=tf.constant_initializer([0.0, 0.0]))
])

model.compile(optimizer='adam', loss='mse', metrics=['accuracy']) 
print(model.weights)
input = np.array([[2.0, 3.0, 2.0]])
output = model.predict(input)
print(output)

tf.saved_model.save(model, '../saved_models/three-in_two-in_nnet')
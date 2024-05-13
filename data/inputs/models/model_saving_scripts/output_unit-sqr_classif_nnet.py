import tensorflow as tf
from tensorflow import keras
import numpy as np

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=4, activation='relu', input_shape=(2,), # (0,0), (0,1), (1,0), (1,1)
                          kernel_initializer=tf.constant_initializer([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]]),
                          bias_initializer=tf.constant_initializer([-1.0, 0.0, -1.0, 0.0])),
    tf.keras.layers.Dense(units=1, activation='sigmoid', # output corresponds to [1 for in square, 0 for not]
                          kernel_initializer=tf.constant_initializer([[-1.0], [-1.0], [-1.0], [-1.0]]),
                          bias_initializer=tf.constant_initializer([0.0]))
])

model.compile(optimizer='adam', loss='mse', metrics=['accuracy']) 

input = np.array([[1.1, 1]])
output = model.predict(input)
print(output)

tf.saved_model.save(model, '../saved_models/unit-sqr_classif_nnet')
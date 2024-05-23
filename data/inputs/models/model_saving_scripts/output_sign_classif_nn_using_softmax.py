import tensorflow as tf
from tensorflow import keras
import numpy as np

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=2, activation='relu', input_shape=(1,),
                          kernel_initializer=tf.constant_initializer([[1.0], [-1.0]]),
                          bias_initializer=tf.constant_initializer([0.0, 0.0])),
    tf.keras.layers.Dense(units=2, activation='softmax', # output corresponds to [positive probability, negative probability]
                          kernel_initializer=tf.constant_initializer([[1.0, 0.0], [0.0, 1.0]]),
                          bias_initializer=tf.constant_initializer([0.0])),
])

model.compile(optimizer='adam', loss='mse', metrics=['accuracy']) 

input = np.array([-1.0])
output = model.predict(input)
print(output)

tf.saved_model.save(model, 'data/inputs/models/saved_models/sign_classif_nn_using_softmax')
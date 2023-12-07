import tensorflow as tf
from tensorflow import keras
import numpy as np

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=2, activation='relu', input_shape=(1,),
                          kernel_initializer=tf.constant_initializer([[1.0], [-1.0]]),
                          bias_initializer=tf.constant_initializer([0.0, 0.0])),
    # tf.keras.layers.Dense(units=2, activation='softmax', # output corresponds to [positive probability, negative probability]
    #                       kernel_initializer=tf.constant_initializer([[1.0, 0.0], [0.0, 1.0]]),
    #                       bias_initializer=tf.constant_initializer([0.0])),

    # tf.keras.layers.Dense(units=2, activation='tanh', # one substitute for softmax
    #                       kernel_initializer=tf.constant_initializer([[1.0, 0.0], [0.0, 1.0]]),
    #                       bias_initializer=tf.constant_initializer([0.0])),
    # tf.keras.layers.Dense(units=1, activation='linear', # to bring down to one output neuron
    #                       kernel_initializer=tf.constant_initializer([[0.5, 0.5]]),
    #                       bias_initializer=tf.constant_initializer([0.5])),

    # tf.keras.layers.Dense(units=2, activation='hard_sigmoid', # a piece-wise linear substitute for softmax (to map values to at most 1)
    #                       kernel_initializer=tf.constant_initializer([[25000.0, 0.0], [0.0, 25000.0]]), # scale most values between 0 and ±2.5 to be more extreme than ±2.5
    #                       bias_initializer=tf.constant_initializer([-2.5])), # ±2.5 is when output becomes 0 or 1
])

model.compile(optimizer='adam', loss='mse', metrics=['accuracy']) 

input = np.array([0.001])
output = model.predict(input)
print(output)

tf.saved_model.save(model, '../saved_models/sign_classif_nn_no-softmax')
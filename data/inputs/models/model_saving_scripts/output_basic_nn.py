import tensorflow as tf
import numpy as np

# 3x + 2y

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=2, activation='relu', input_shape=(2,),
                          kernel_initializer=tf.constant_initializer([[1.0, 0.0], [0.0, 1.0]]),
                          bias_initializer=tf.constant_initializer([0.0, 0.0])),
    tf.keras.layers.Dense(units=1, activation='linear',
                          kernel_initializer=tf.constant_initializer([[3.0, 2.0]]),
                          bias_initializer=tf.constant_initializer([0.0]))
])

model.compile(optimizer='adam', loss='mse', metrics=['accuracy']) 
print("here")
input = np.array([[2.0, 1.0]])
output = model.predict(input)
print(output)

tf.saved_model.save(model, '../saved_models/basic_nn')
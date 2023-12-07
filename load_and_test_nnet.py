import tensorflow as tf
import numpy as np

model_path = "reluplex_fig2_nnet"

loaded = tf.saved_model.load(model_path)
model = loaded.signatures["serving_default"]

input = np.array([100.0])
output = model(input)
print(output)
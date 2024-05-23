import tensorflow as tf
import numpy as np

model_path = "data/inputs/models/saved_models/reluplex_paper_fig2_nn"

# Load the saved model
loaded = tf.saved_model.load(model_path)
model = loaded.signatures["serving_default"]

# Create the input as a TensorFlow tensor
input_tensor = tf.constant([[100.0]])

# Obtain output
output_dict = model(dense_input=input_tensor)
output_array = output_dict['dense_1'].numpy()
extracted_output = output_array[0][0]

print(extracted_output)

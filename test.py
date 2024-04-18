import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model
model = load_model('keras_model.h5')

# Load the class indices from the text file
class_indices_path = 'index.txt'

class_indices = {}
with open(class_indices_path, 'r') as file:
    for line in file:
        key, value = line.strip().split(':')
        class_indices[int(key)] = value

# Define a function to preprocess the uploaded image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize pixel values
    return img_array

# Define a function to make predictions
def predict_crop_disease(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(prediction)
    return predicted_class_index

# Test the model with an uploaded image
uploaded_image_path = r"C:\Users\shilp\OneDrive\Documents\Luminar\Internship\crop disease\dataset\test\Apple___Black_rot\f415bc3e-3e71-4636-a3dd-78b65002384d___JR_FrgE.S 2917_270deg.JPG"
predicted_class_index = predict_crop_disease(uploaded_image_path)
predicted_class_name = class_indices[predicted_class_index]

# Print the predicted class name
print("Predicted class:", predicted_class_name)

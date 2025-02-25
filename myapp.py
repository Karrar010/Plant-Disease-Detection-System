import tensorflow as tf
from tensorflow.keras import models, layers
import streamlit as st
from PIL import Image
import numpy as np
import pdb
import json


model = tf.keras.models.load_model('mymodel.keras')

# Create a Streamlit app
st.title("Plant Disease Detection")
st.write("Upload an image of a Plant Leaf:")

# Create a file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

# Create a button to trigger prediction
predict_button = st.button("Predict")

def preprocess_image(image):
    image = image.resize((256,256))  # Resize to 28x28
    image = np.array(image)  # Convert to numpy array
    if image.ndim == 2:  # If the image is grayscale, convert to RGB
        image = np.stack([image]*3, axis=-1)
    image = image.reshape(1, 256, 256, 3)  # Reshape to match input shape
    image = image.astype('float32') / 255  # Normalize
    return image

# Load class names
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# Define a function to predict the disease
def predict_disease(image):
    prediction = model.predict(image)
    disease = class_names[np.argmax(prediction)]
    return disease

## Create a main function to run the app
def main():
    prediction_output = st.empty()  # Define prediction_output as a placeholder
    if uploaded_file:
        image = Image.open(uploaded_file)
        preprocessed_image = preprocess_image(image)
        if predict_button:
            disease = predict_disease(preprocessed_image)
            prediction_output.text(f"The predicted disease is: {disease}")  # Update the placeholder with the prediction result


# Run the app
if __name__ == "__main__":
    main()

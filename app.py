%%writefile app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("/content/drive/MyDrive/images/my_model.h5")

   # Define image size and labels
image_size = (128, 128)
labels = ['Angry', 'Happy', 'Relaxed', 'Sad']

   # Define the Streamlit app
st.title("Emotion Recognition App")

   # Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
  image = Image.open(uploaded_file).convert('RGB')
  image = image.resize(image_size)
  image_array = np.array(image) / 255.0
  image_array = np.expand_dims(image_array, axis=0)

       # Make a prediction
  prediction = model.predict(image_array)
  predicted_class_index = np.argmax(prediction)
  predicted_emotion = labels[predicted_class_index]

       # Display the results
  st.image(image, caption="Uploaded Image", use_container_width=True)
  st.write(f"Predicted Emotion: **{predicted_emotion}**")

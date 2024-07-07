import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf


# load model.h5
model = tf.keras.models.load_model('./cat_dog.h5')

st.title("Cat or Dog?")

# image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
# show uploaded picture
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # prepare image for model
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0 

    # prediction
    prediction = model.predict(image)
    confidence = np.max(prediction) * 100 
    class_label = "Cat" if np.argmax(prediction) == 0 else "Dog"

    st.write(f"This is a {class_label} with {confidence:.2f}% confidence.")
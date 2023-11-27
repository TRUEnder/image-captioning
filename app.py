import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf

# Image Preprocessing

def preprocess_image(image):
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.keras.layers.Resizing(299, 299)(image)
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image

# Call Model

def generate_caption(image, add_noise=False):
    caption_model = tf.keras.models.load_model('MODEL_FILE')
    img = preprocess_image(image)


# GUI

st.title('Image Captioning App')
st.write('Using Inception V3 model and transformer')

uploaded_file = st.file_uploader("Insert an image")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image)

    # caption = generate_caption(image)
    st.write('Caption : Blue car across the sandy road')
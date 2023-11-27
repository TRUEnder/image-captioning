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
    # image = tf.keras.applications.inception_resnet_v2.preprocess_input(img)
    return image

# Call Model

def generate_caption(image, add_noise=False):
    caption_model = tf.keras.models.load_model('MODEL_FILE')
    img = preprocess_image(image)
    
    # if add_noise:
    #     noise = tf.random.normal(img.shape)*0.1
    #     img = img + noise
    #     img = (img - tf.reduce_min(img))/(tf.reduce_max(img) - tf.reduce_min(img))
    
    # img = tf.expand_dims(img, axis=0)
    # img_embed = caption_model.cnn_model(img)
    # img_encoded = caption_model.encoder(img_embed, training=False)

    # y_inp = '[start]'
    # for i in range(MAX_LENGTH-1):
    #     tokenized = tokenizer([y_inp])[:, :-1]
    #     mask = tf.cast(tokenized != 0, tf.int32)
    #     pred = caption_model.decoder(
    #         tokenized, img_encoded, training=False, mask=mask)
        
    #     pred_idx = np.argmax(pred[0, i, :])
    #     pred_idx = tf.convert_to_tensor(pred_idx)
    #     pred_word = idx2word(pred_idx).numpy().decode('utf-8')
    #     if pred_word == '[end]':
    #         break
        
    #     y_inp += ' ' + pred_word
    
    # y_inp = y_inp.replace('[start] ', '')
    # return y_inp


# GUI

st.title('Image Captioning App')
st.write('Using Inception V3 model and transformer')

uploaded_file = st.file_uploader("Insert an image")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image)

    # caption = generate_caption(image)
    st.write('Caption : Blue car jump inside a woman hoe')
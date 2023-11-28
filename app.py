import numpy as np
import streamlit as st
import tensorflow as tf
import pickle
from PIL import Image

def preprocess(input_image):
    img = input_image.resize((299, 299))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.inception_v3.preprocess_input(x)
    return x

def encode(input_image):
    image = preprocess(input_image)

    inceptionv3 = tf.keras.models.load_model('inception_v3.h5')

    fea_vec = inceptionv3.predict(image)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec

def read_binary(filename):
    # for reading binary file
    with open(filename, 'rb') as fp:
        obj = pickle.load(fp)
        return obj

def beam_search_predictions(image, beam_index = 3):
    model = tf.keras.models.load_model('image_captioning_model.h5')
    wordtoix = read_binary('wordtoix')
    ixtoword = read_binary('ixtoword')
    max_length = 51

    start = [wordtoix["startseq"]]
    start_word = [[start, 0.0]]
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            par_caps = tf.keras.preprocessing.sequence.pad_sequences([s[0]], maxlen=max_length, padding='post')
            preds = model.predict([image,par_caps], verbose=0)
            word_preds = np.argsort(preds[0])[-beam_index:]
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [ixtoword[i] for i in start_word]
    final_caption = []
    
    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption

# Interface

def generate_caption(input_image):
    image = encode(input_image)
    image = image.reshape((1, 2048))
    return beam_search_predictions(image, beam_index=3)

# GUI

st.title('Image Captioning App')
st.write('Using Inception V3 model and LSTM')

uploaded_file = st.file_uploader("Insert an image")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image)

    with st.spinner('Wait for it...') :
        caption = generate_caption(image)

    if (caption != None) :
        st.divider()
        st.success('This is a success message!', icon="✅")
        st.write('Caption')
        st.code(caption, language='bash')
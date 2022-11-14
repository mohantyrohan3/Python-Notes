import streamlit as st
import numpy as np
import matplotlib as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image


classes=['Boot','Sandel','Shoe']

model = keras.models.load_model('models/{1}')

def predict(img,model):
    image = tf.keras.preprocessing.image.load_img(img)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    return classes[np.argmax(predictions[0])]


st.write(""" 
    # Shoe vs Sandle vs Boot Classification

 """)

file=st.file_uploader("Please Upload an Image ",type=['jpg','png'])

if file is None:
    st.text("Please upload an valid image file ")
else:
    ans=predict(file,model)
    image1 = tf.keras.preprocessing.image.load_img(file)
    st.image(image1,use_column_width=True)
    st.success("The above image is "+ ans)


import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.utils import img_to_array
from PIL import Image
import io

def load_image():
    uploaded_file = st.file_uploader(label='')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data,,width = 200)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

def load_model():
	model = tf.keras.models.load_model("HCR-CNN.h5")
	return model


def main():
    st.title('Hindi Character Recognition')
    labels =['yna', 't`aa', 't`haa', 'd`aa', 'd`haa', 'a`dna', 'ta', 'tha', 'da', 'dha', 'ka', 'na', 'pa', 'pha', 'ba', 'bha', 'ma', 'yaw', 'ra', 'la', 'waw', 'kha', 'sha', 'shat', 'sa', 'ha', 'aksha', 'tra', 'gya', 'ga', 'gha', 'kna', 'cha', 'chha', 'ja', 'jha', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    st.subheader("Please upload you hindi character image to predict")
    up_image = load_image()
    model = load_model()
    if up_image:
        image = img_to_array(up_image)
        image = cv2.resize(image, (32,32))
        image = image.astype("float")/255.0
        print(image.shape)
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=3)
    result = st.button('Predict')
    if result:
        lists = model.predict(image)[0]
        Predicted_character = labels[np.argmax(lists)]
        st.write('Calculating results...')
        st.write('Predcited character: ',Predicted_character)

if __name__ == '__main__':
    main()

	

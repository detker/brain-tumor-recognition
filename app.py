###################################################
# author: Wojciech Krolikowski (@detker)
# date: 29.06.2023
# version: 1.0
# about: web application presenting a deep-learning 
#          neural network predicting a brain tumor, 
#           pre-trained with EfficientNetB0, 
#           created with streamlit help.
###################################################

import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
from skimage import io
import cv2

# decorator for storing model data in cache
@st.cache_resource()
# loads the model
def load_model_f():
    model = load_model('model-5.h5')
    
    return model

# preprocess uploaded user image
def preprocess_img(uploaded_img):
    pic = np.array(io.imread(uploaded_img))
    pic_resized = cv2.resize(pic, (150, 150))
    if len(pic_resized) == 150*150:
        help = np.zeros((150, 150, 3))
        help[:, :, 0] = pic_resized
        help[:, :, 1] = pic_resized
        help[:, :, 2] = pic_resized
        pic_resized = help
    pic_resized = pic_resized.reshape(1, 150, 150, 3)
    
    return pic_resized

# formats and presents the results
def show_results(y_hat, labels):
    result = labels[np.argmax(y_hat)]
    
    st.subheader('model prediction: `{}`'.format(result))
    st.write('')
    
    df = pd.DataFrame(y_hat, index=labels, columns=['probability'])
    df.index.name = 'type of tumor'
    df['probability'] = df['probability'].apply(lambda x: round(x, 1_000_000))
    
    with st.expander('##### expand for prediction details:'):
        st.dataframe(df)

if __name__ == '__main__':
    st.title('brain tumor recognition')
    st.markdown("*upload human brain's x-ray scan. trained neural network will try to recognize whether tumor is present or not. if so, the type of tumor will also be predicted. model includes pre-trained on imagenet dataset wages from efficient_net. accuracy level above 98%. enjoy!*")
    st.markdown('*dekter, 2023* | `github.com/detker`')
    
    uploaded_img = st.file_uploader('upload x-ray scan:', type=['jpg', 'jpeg', 'png', 'bmp'])
    model = load_model_f()
    labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

    if uploaded_img:
        left_col, center_col, right_col = st.columns(3)
        with center_col:
            st.image(uploaded_img, caption='uploaded x-ray', width=200)
        preprocessed_img = preprocess_img(uploaded_img)
        y_hat = model.predict(preprocessed_img)[0]
        show_results(y_hat, labels)

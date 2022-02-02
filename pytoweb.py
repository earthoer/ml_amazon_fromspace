
import streamlit as st
from PIL import Image
import cv2 as cv
import tensorflow as tf
import  keras
import pandas as pd
import numpy as np
import keras.applications
from tensorflow import keras
# st.set_page_config(layout="wide")
def f2_score(y_true, y_pred):
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(tf.round(y_pred), "int32")
    y_correct = y_true * y_pred
    sum_true = tf.reduce_sum(y_true, axis=1)
    sum_pred = tf.reduce_sum(y_pred, axis=1)
    sum_correct = tf.reduce_sum(y_correct, axis=1)
    precision = sum_correct / sum_pred
    recall = sum_correct / sum_true
    f_score = 5 * precision * recall / (4 * precision + recall)
    f_score = tf.where(tf.math.is_nan(f_score), tf.zeros_like(f_score), f_score)
    return tf.reduce_mean(f_score)
    base_config = super().get_config()
    return {**base_config, "num_classes": self.num_classes}
def main():
    new_title = '<p style="font-family:showcard gothic; color:White; text-align:center;font-size: 80px;"> Planet: Understanding the Amazon from Space </p>'
    st.markdown(new_title, unsafe_allow_html=True)
    model = keras.models.load_model('saved_model\my_model', custom_objects=None, compile=False, options=None)
    st.write("Upload Photo:")
    image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
    label = ['agriculture','artisinal_mine','bare_ground','blooming', 'blow_down','clear','cloudy','conventional_mine','cultivation','habitation','haze','partly_cloudy','primary','road','selective_logging','slash_burn','water']
    if(image_file is not None):
        
        image = Image.open(image_file)
        cv_image = np.array(image.convert('RGB'))
        cv_image = cv.cvtColor(cv_image,1)
        st.image(cv_image, caption='Input', use_column_width=True,width=1500)
        image = cv.resize(cv_image,(128,128))
        image = image/255
        image = np.expand_dims(image,axis=0)
        predict = model.predict(image)
        new_title = '<p style="font-family:showcard gothic; color:White; font-size: 42px;">predict answer is :</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        # st.write("predict answer is :")
        
        for i in range(len(predict[0])):
            if predict[0][i] > 0.2:
                new_title = '<p style="font-family:showcard gothic; color:#87CEEB; font-size: 25px;">'+label[i]+'</p>'
                st.markdown(new_title, unsafe_allow_html=True)
        
        
        
if __name__ == '__main__':
        main()
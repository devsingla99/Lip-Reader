# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 09:36:34 2021

@author: DEVANSHU
"""

import streamlit as st
from PIL import Image

st.title("         LIP READER \n \n \n")
st.sidebar.header("LIP READER \n")
st.sidebar.write("This application predicts words and phrases by identifying lip movement")
st.sidebar.header("Instructions.. \n")
st.sidebar.write("Your face must be clearly visible in camera")
st.sidebar.write("You must not wear shades, mask or any other equipment that covers eyes, nose and lips")
st.write("Words that can be predicted")
image=Image.open(r'C:\Users\DEVANSHU\Desktop\Projects\DeepLearning\lip model training\pictures\category.png')
st.image(image,width=600)
st.title("\n \n")
result=st.button("Start Prediction")
if result:
    import test.py
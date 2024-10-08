import os
import streamlit as st
import cv2 as cv
import numpy as np

from preprocess_dataset import IMAGE_PATH, HIST_PATH, load_hist
from retrieve_images import retrieve_similar_images


hist_lists = load_hist(HIST_PATH)
image_paths = []
for root, _, files in os.walk(IMAGE_PATH):
    for file in files:
        image_paths.append(f"{root}//{file}")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv.imdecode(file_bytes, 1)
    
    st.image(image, caption="Uploaded Image.", width=300)

number_of_images = st.number_input(
    "Choose a number", 
    min_value=0,
    max_value=50,
    value=10,
    step=5
)


if st.button("Retrieve"):
    if uploaded_file is None:
        st.markdown("<hr style='border: 2px solid red;'>", unsafe_allow_html=True)
        st.error("Please input an image")
    else:
        grayscale_image = cv.imdecode(file_bytes, cv.IMREAD_GRAYSCALE)
        retrieved_indexes = retrieve_similar_images(grayscale_image, hist_lists, number_of_images)

        retrieved_images = [image_paths[index] for index in retrieved_indexes]

        st.write("Retrieved images...")
        for image in retrieved_images:
            st.image(image, width=300)

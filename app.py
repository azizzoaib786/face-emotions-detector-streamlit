import streamlit as st
from PIL import Image
import numpy as np
from deepface import DeepFace
import cv2
import os

# Optional: Suppress PyAV warning if needed
os.environ["PYAV_LOGLEVEL"] = "error"

st.set_page_config(page_title="😊 Face Emotion Detector", layout="centered")
st.title("😊 Face Emotion Detector")

# Select mode
option = st.radio("Choose Input Mode:", ["Upload Image", "Take a Photo"])

# Store captured image
if "captured_image" not in st.session_state:
    st.session_state.captured_image = None

# Upload image
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Analyzing emotions...")

        try:
            result = DeepFace.analyze(image_np, actions=['emotion'], enforce_detection=False)
            st.success(f"Detected emotion: {result[0]['dominant_emotion'].capitalize()}")
        except Exception as e:
            st.error(f"Error detecting emotion: {str(e)}")

# Capture from webcam
elif option == "Take a Photo":
    picture = st.camera_input("Take a picture")

    if picture is not None:
        image = Image.open(picture).convert("RGB")
        image_np = np.array(image)

        st.image(image, caption="Captured Image", use_column_width=True)
        st.write("Analyzing emotions...")

        try:
            result = DeepFace.analyze(image_np, actions=['emotion'], enforce_detection=False)
            st.success(f"Detected emotion: {result[0]['dominant_emotion'].capitalize()}")
        except Exception as e:
            st.error(f"Error detecting emotion: {str(e)}")

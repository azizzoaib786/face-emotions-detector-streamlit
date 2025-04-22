import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av
import numpy as np
from deepface import DeepFace
from PIL import Image
import cv2
import os

os.environ["PYAV_LOGLEVEL"] = "error"

st.set_page_config(page_title="Face Emotion Detector", layout="centered")
st.title("😊 Face Emotion Detection")

# Select mode
option = st.radio("Choose Input Mode:", ["Upload Image", "Take a Photo"])

# Store captured image
if "captured_image" not in st.session_state:
    st.session_state.captured_image = None

# Upload image
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Analyzing..."):
            try:
                result = DeepFace.analyze(image_np, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                st.success(f"Detected Emotion: {emotion.capitalize()}")
                st.subheader("Emotion Scores:")
                st.json(result[0]['emotion'])
            except Exception as e:
                st.error(f"Analysis Error: {e}")

# Cam mode
elif option == "Take a Photo":
    class SnapshotTransformer(VideoTransformerBase):
        def __init__(self):
            self.frame = None

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            img = frame.to_ndarray(format="bgr24")
            self.frame = img.copy()
            return img

    ctx = webrtc_streamer(
        key="snapshot",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=SnapshotTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_transformer:
        if st.button("📸 Take Picture"):
            st.session_state.captured_image = ctx.video_transformer.frame
            st.success("Picture captured!")

    if st.session_state.captured_image is not None:
        captured_img = st.session_state.captured_image
        st.image(cv2.cvtColor(captured_img, cv2.COLOR_BGR2RGB), caption="Captured Image", use_column_width=True)

        with st.spinner("Analyzing..."):
            try:
                result = DeepFace.analyze(captured_img, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                st.success(f"Detected Emotion: {emotion.capitalize()}")
                st.subheader("Emotion Scores:")
                st.json(result[0]['emotion'])
            except Exception as e:
                st.error(f"Analysis Error: {e}")
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av
import numpy as np
from deepface import DeepFace
import cv2
import os

os.environ["PYAV_LOGLEVEL"] = "error"

st.set_page_config(page_title="Face Emotion Detector", layout="centered")
st.title("😊 Real-time Face Emotion Detection")

# Select mode
option = st.radio("Choose Input Mode:", ["Upload Image", "Real-time Webcam"])

# Upload image
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_np = np.array(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1))
        st.image(image_np, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Analyzing..."):
            try:
                result = DeepFace.analyze(image_np, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                st.success(f"Detected Emotion: {emotion.capitalize()}")
                st.subheader("Emotion Scores:")
                st.json(result[0]['emotion'])
            except Exception as e:
                st.error(f"Analysis Error: {e}")

# Real-time mode
elif option == "Real-time Webcam":
    class EmotionTransformer(VideoTransformerBase):
        def __init__(self):
            self.frame_count = 0
            self.last_emotion = None
            self.last_scores = None

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1

            if self.frame_count % 15 == 0:  # Process every 15th frame
                try:
                    result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
                    self.last_emotion = result[0]['dominant_emotion']
                    self.last_scores = result[0]['emotion']
                except Exception:
                    self.last_emotion = "Unknown"
                    self.last_scores = {}

            if self.last_emotion:
                cv2.putText(img, f"Emotion: {self.last_emotion}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            return img

    webrtc_streamer(
        key="realtime",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=EmotionTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.info("Model processes emotion every 15 frames. Best viewed on Chrome.")
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av
import cv2
import numpy as np
import insightface

# Load InsightFace model
@st.cache_resource
def load_model():
    model = insightface.app.FaceAnalysis(name="buffalo_l")
    model.prepare(ctx_id=0)  # CPU mode
    return model

face_model = load_model()

# Emotion emoji map
emotion_emojis = {
    'neutral': '😐',
    'happy': '😄',
    'sad': '😢',
    'surprise': '😲',
    'fear': '😨',
    'angry': '😠',
    'disgust': '🤢',
    'unknown': '❓'
}

class EmotionDetector(VideoTransformerBase):
    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_model.get(rgb)

        for face in faces:
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box
            emotion = face.emotion

            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label emotion + emoji
            label = f"{emotion.capitalize()} {emotion_emojis.get(emotion, '')}"
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        return img

# Streamlit App
st.set_page_config(page_title="😊 Real-time Face Emotion Detector", layout="centered")
st.title("😊 Real-time Face Emotion Detection (InsightFace)")

webrtc_streamer(
    key="emotion",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=EmotionDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.info("Emotions detected real-time using InsightFace models 🚀")
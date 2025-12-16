import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import json
import os
import gdown

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(page_title="Team Face Recognition", layout="centered")
st.title("👤 Team Face Recognition (Live Webcam)")

# -------------------------------
# Download model from Google Drive if not present
# -------------------------------
MODEL_PATH = "face_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=116t-OB2Dc5w78aBwoWkfgj0sEOi0Ro5Q"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model... Please wait ⏳")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# -------------------------------
# Load model
# -------------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# -------------------------------
# Load labels
# -------------------------------
with open("labels.json", "r") as f:
    class_indices = json.load(f)

labels = list(class_indices.keys())

# -------------------------------
# Load Haar Cascade for face detection
# -------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------------------
# Webcam video processor
# -------------------------------
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5
        )

        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, (160, 160))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)

            prediction = model.predict(face, verbose=0)
            predicted_label = labels[np.argmax(prediction)]

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                img,
                predicted_label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        return img

# -------------------------------
# Start webcam
# -------------------------------
webrtc_streamer(
    key="face-recognition",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

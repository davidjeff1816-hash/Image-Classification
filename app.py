import streamlit as st
import cv2
import numpy as np
import json
import os
import gdown
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ------------------------------------
# Streamlit Page Config
# ------------------------------------
st.set_page_config(page_title="Team Face Recognition", layout="centered")
st.title("👤 Team Face Recognition (Live Webcam)")

# ------------------------------------
# Download TFLite model from Google Drive
# ------------------------------------
MODEL_PATH = "face_model.tflite"
MODEL_URL = "https://drive.google.com/uc?id=1Ce-wB10lyAIEQ5Ht-oDIQEdLEEeBvPJv"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading face recognition model… ⏳")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ------------------------------------
# Load TFLite model
# ------------------------------------
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ------------------------------------
# Load class labels
# ------------------------------------
with open("labels.json", "r") as f:
    class_indices = json.load(f)

labels = list(class_indices.keys())

# ------------------------------------
# Load Haar Cascade Face Detector
# ------------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ------------------------------------
# Prediction function
# ------------------------------------
def predict_face(face_img):
    interpreter.set_tensor(
        input_details[0]['index'],
        face_img.astype(np.float32)
    )
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

# ------------------------------------
# Webcam Video Processor
# ------------------------------------
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, (160, 160))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)

            prediction = predict_face(face)
            label = labels[np.argmax(prediction)]

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                img,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        return img

# ------------------------------------
# Start Webcam Stream
# ------------------------------------
webrtc_streamer(
    key="face-recognition",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

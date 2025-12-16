import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import json
import os
import gdown
from PIL import Image

st.set_page_config(page_title="Team Face Recognition", layout="centered")
st.title("👤 Team Face Recognition")

# -------------------------
# Download TFLite model
# -------------------------
MODEL_PATH = "face_model.tflite"
MODEL_URL = "https://drive.google.com/uc?id=1Ce-wB10lyAIEQ5Ht-oDIQEdLEEeBvPJv"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# -------------------------
# Load TFLite model
# -------------------------
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -------------------------
# Load labels
# -------------------------
with open("labels.json") as f:
    labels = list(json.load(f).keys())

# -------------------------
# Face detector
# -------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def predict(face):
    interpreter.set_tensor(input_details[0]['index'], face.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

# -------------------------
# Upload image
# -------------------------
uploaded = st.file_uploader("Upload a face image", type=["jpg","png","jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face = img_np[y:y+h, x:x+w]
        face = cv2.resize(face, (160,160))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)

        pred = predict(face)
        label = labels[np.argmax(pred)]

        cv2.rectangle(img_np, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(img_np, label, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    st.image(img_np, caption="Prediction", use_column_width=True)

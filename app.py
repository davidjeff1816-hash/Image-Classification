import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import json

# ----------------------------------
# Streamlit Page Setup
# ----------------------------------
st.set_page_config(page_title="Team Face Recognition", layout="centered")
st.title("👤 Team Face Recognition (Live Webcam)")

# ----------------------------------
# Load TFLite Model
# ----------------------------------
interpreter = tf.lite.Interpreter(model_path="face_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ----------------------------------
# Load Labels (Names)
# ----------------------------------
with open("labels.json", "r") as f:
    class_indices = json.load(f)

labels = list(class_indices.keys())  # ['jeff', 'arun', 'rahul']

# ----------------------------------
# Load Face Detector
# ----------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ----------------------------------
# Prediction Function
# ----------------------------------
def predict_face(face_img):
    interpreter.set_tensor(
        input_details[0]['index'],
        face_img.astype(np.float32)
    )
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

# ----------------------------------
# Webcam Control
# ----------------------------------
run = st.checkbox("Start Webcam")
frame_window = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("❌ Unable to access webcam")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)

        prediction = predict_face(face)

        pred_index = np.argmax(prediction)
        confidence = prediction[0][pred_index] * 100
        person_name = labels[pred_index]

        label_text = f"{person_name} ({confidence:.1f}%)"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label_text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    frame_window.image(frame, channels="BGR")

cap.release()

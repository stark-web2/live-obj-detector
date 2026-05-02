import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="YOLO Object Detection", layout="wide")

st.title("🎥 Simple YOLO Object Detection (Deployment Safe)")
st.write("Take a picture using your camera and detect objects.")

# Load model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()


img_file = st.camera_input("Capture Image")

if img_file is not None:
    
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    results = model(frame)
    annotated = results[0].plot()


    st.image(annotated, channels="BGR", caption="Detected Objects")

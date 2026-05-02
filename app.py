import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2
from collections import Counter

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Live Object Detection & Tracing", layout="wide")

st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to detect, track, and count them in real-time.")

# ----------------------------
# LOAD MODEL (cached)
# ----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ----------------------------
# VIDEO PROCESSING FUNCTION
# ----------------------------
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # YOLOv8 tracking
    results = model.track(
        img,
        persist=True,
        conf=0.5,
        verbose=False
    )

    annotated_frame = results[0].plot()

    # ----------------------------
    # OBJECT COUNTING + LABELS
    # ----------------------------
    names = results[0].names
    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        class_ids = boxes.cls.cpu().numpy().astype(int)
        counts = Counter(class_ids)

        y_offset = 30

        for cls_id, count in counts.items():
            label = names[cls_id]

            cv2.putText(
                annotated_frame,
                f"{label}: {count}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            y_offset += 30

        # ----------------------------
        # ALERT: PERSON DETECTED
        # ----------------------------
        detected_labels = [names[i] for i in class_ids]

        if "person" in detected_labels:
            cv2.putText(
                annotated_frame,
                "⚠ PERSON DETECTED",
                (10, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


# ----------------------------
# START WEBCAM STREAM
# ----------------------------
webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    async_processing=False,   # FIX: more stable camera
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)

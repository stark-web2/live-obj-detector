import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
from ultralytics import YOLO
from collections import defaultdict
import time

st.set_page_config(page_title="YOLO Live Camera", layout="wide")

st.title("🎥 Live Object Detection & Tracing (WebRTC FIXED)")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Sidebar controls
st.sidebar.header("⚙️ Controls")
conf = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
alert_object = st.sidebar.text_input("🔔 Alert Object")

object_counts = defaultdict(int)
detection_log = []

# ---------------------------
# YOLO Video Transformer
# ---------------------------
class YOLOTransformer(VideoTransformerBase):
    def transform(self, frame: av.VideoFrame):

        img = frame.to_ndarray(format="bgr24")

        # YOLO inference
        results = model.track(img, persist=True, conf=conf)
        annotated = img.copy()

        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf_score = float(box.conf[0])
                label = model.names[cls]

                # Draw box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"{label} {conf_score:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

                # Alert system
                if alert_object and label.lower() == alert_object.lower():
                    cv2.putText(
                        annotated,
                        "ALERT!",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3
                    )

                detection_log.append(
                    f"{time.strftime('%H:%M:%S')} - {label} ({conf_score:.2f})"
                )

        return annotated


webrtc_streamer(
    key="yolo-camera",
    video_transformer_factory=YOLOTransformer,
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)

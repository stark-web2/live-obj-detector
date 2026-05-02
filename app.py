import streamlit as st
import cv2
import time
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import os
import threading
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.set_page_config(page_title="Live Object Detection & Tracing", layout="wide")

st.markdown("""
<style>
body { background-color: #0e1117; color: white; }
.stApp { background-color: #0e1117; }
</style>
""", unsafe_allow_html=True)

st.title("🎥 Live Object Detection & Tracing")
st.markdown("### 🚀 AI-Powered Real-Time Detection System")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Initialize session state
if 'object_counts' not in st.session_state:
    st.session_state.object_counts = defaultdict(int)
if 'tracked_ids' not in st.session_state:
    st.session_state.tracked_ids = set()
if 'detection_log' not in st.session_state:
    st.session_state.detection_log = []
if 'saved_images' not in st.session_state:
    st.session_state.saved_images = []

# Create saved_frames directory
if not os.path.exists("saved_frames"):
    os.makedirs("saved_frames")

st.sidebar.header("⚙️ Controls")
conf = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
alert_object = st.sidebar.text_input("🔔 Alert Object")
save_frames = st.sidebar.checkbox("💾 Save Frames")
show_logs = st.sidebar.checkbox("📝 Show Detection Log", value=True)

# Object counts display
st.sidebar.subheader("📊 Object Counts")
for obj, count in dict(st.session_state.object_counts).items():
    st.sidebar.write(f"{obj}: {count}")

class VideoProcessor:
    def __init__(self):
        self.model = model
        self.conf = conf
        self.alert_object = alert_object
        self.save_frames = save_frames
        self.frame_count = 0
        self.prev_time = time.time()
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        self.frame_count += 1
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        
        # Run YOLO detection
        results = self.model.track(img, persist=True, conf=self.conf, verbose=False)
        annotated = img.copy()
        
        if results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf_score = float(box.conf[0])
                label = self.model.names[cls]
                
                # Draw bounding box and label
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"{label} ({conf_score:.2f})", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Track object IDs
                if box.id is not None:
                    obj_id = int(box.id[0])
                    if obj_id not in st.session_state.tracked_ids:
                        st.session_state.tracked_ids.add(obj_id)
                        st.session_state.object_counts[label] += 1
                
                # Alert system
                if self.alert_object and label.lower() == self.alert_object.lower():
                    st.session_state.detection_log.append(
                        f"🚨 ALERT {time.strftime('%H:%M:%S')} - {label} ({conf_score:.2f})"
                    )
                
                # Log detection
                st.session_state.detection_log.append(
                    f"{time.strftime('%H:%M:%S')} - {label} ({conf_score:.2f})"
                )
        
        # Save frames
        if self.save_frames and self.frame_count % 30 == 0:
            filename = f"saved_frames/frame_{int(time.time())}.jpg"
            cv2.imwrite(filename, annotated)
            st.session_state.saved_images.append(filename)
        
        # Update FPS display
        st.session_state.fps = fps
        
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    st.header("📷 Live Camera Feed")
    RTC_CONFIGURATION = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })
    
    webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 30}
            }
        }
    )

with col2:
    if 'fps' in st.session_state:
        st.metric("⚡ FPS", f"{st.session_state.fps:.1f}")
    
    st.subheader("🔍 Status")
    st.success("✅ Camera Active")
    if alert_object:
        st.info(f"🔔 Monitoring: **{alert_object}**")

# Saved images download
st.markdown("---")
st.header("💾 Download Saved Frames")
if st.session_state.saved_images:
    for img_path in st.session_state.saved_images[-5:]:  # Show last 5
        with open(img_path, "rb") as file:
            st.download_button(
                label=f"📥 {os.path.basename(img_path)}",
                data=file,
                file_name=os.path.basename(img_path),
                mime="image/jpeg",
                key=f"download_{img_path}"
            )
else:
    st.info("No saved frames yet. Enable 'Save Frames' to capture.")

# Detection log
st.markdown("---")
st.header("📄 Detection Log")
if show_logs and st.session_state.detection_log:
    st.text_area("Recent Detections", 
                "\n".join(st.session_state.detection_log[-20:]), 
                height=200)
else:
    st.info("No detections logged yet.")

# Reflection section
st.markdown("---")
st.header("💭 Analysis")
st.markdown("""
### Key Questions:
- **What objects were easily detected?**
- **What affects detection accuracy?**
  - Lighting conditions
  - Object distance/motion
  - Camera quality
  - Background complexity
""")

import streamlit as st
import cv2
import time
from ultralytics import YOLO
from collections import defaultdict
import os

st.set_page_config(page_title="Live Object Detection & Tracing", layout="wide")

st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
.stApp {
    background-color: #0e1117;
}
</style>
""", unsafe_allow_html=True)

st.title("🎥 Live Object Detection & Tracing")
st.markdown("### 🚀 AI-Powered Real-Time Detection System")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()


st.sidebar.header("⚙️ Controls")

conf = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
run = st.sidebar.checkbox("Start Camera")

alert_object = st.sidebar.text_input("🔔 Alert Object")
save_frames = st.sidebar.checkbox("💾 Save Frames")
show_logs = st.sidebar.checkbox("📝 Show Detection Log")


object_counts = defaultdict(int)
tracked_ids = set()
detection_log = []
saved_images = []


if not os.path.exists("saved_frames"):
    os.makedirs("saved_frames")

frame_placeholder = st.empty()
status_text = st.empty()
fps_text = st.empty()

camera = cv2.VideoCapture(0)
frame_count = 0
prev_time = time.time()

if run:
    status_text.success("📷 Camera running...")

    while True:
        ret, frame = camera.read()
        if not ret:
            st.error("❌ Camera not detected")
            break

        frame_count += 1

       
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

      
        results = model.track(frame, persist=True, conf=conf)

        annotated = frame.copy()

        if results[0].boxes is not None:
            boxes = results[0].boxes

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf_score = float(box.conf[0])
                label = model.names[cls]

                
                text = f"{label} ({conf_score:.2f})"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(annotated, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

           
                if box.id is not None:
                    obj_id = int(box.id[0])

                    if obj_id not in tracked_ids:
                        tracked_ids.add(obj_id)
                        object_counts[label] += 1

                # ALERT SYSTEM
                if alert_object and label.lower() == alert_object.lower():
                    st.warning(f"⚠️ ALERT: {label} detected!")

           
                detection_log.append(f"{time.strftime('%H:%M:%S')} - {label} ({conf_score:.2f})")

    
        if save_frames and frame_count % 30 == 0:
            filename = f"saved_frames/frame_{int(time.time())}.jpg"
            cv2.imwrite(filename, annotated)
            saved_images.append(filename)


        frame_placeholder.image(annotated, channels="BGR")

       
        fps_text.markdown(f"### ⚡ FPS: {fps:.2f}")


        with st.sidebar:
            st.subheader("📊 Object Counts")
            for obj, count in object_counts.items():
                st.write(f"{obj}: {count}")

        # Stop condition fix
        if not st.session_state.get("Start Camera", True):
            break

else:
    status_text.warning("⚠️ Camera OFF")

camera.release()


st.markdown("---")
st.header("💾 Download Saved Frames")

if saved_images:
    for img_path in saved_images:
        with open(img_path, "rb") as file:
            st.download_button(
                label=f"Download {os.path.basename(img_path)}",
                data=file,
                file_name=os.path.basename(img_path),
                mime="image/jpeg"
            )
else:
    st.write("No saved frames yet.")

st.markdown("---")
st.header("📄 Observation & Report")

if show_logs:
    st.subheader("📝 Detection Log")
    st.write(detection_log[-20:] if detection_log else "No detections yet.")

st.subheader("💭 Reflection Questions")
st.markdown("""
- What objects were easily detected?  
- What affects detection accuracy?  
  - Lighting  
  - Motion  
  - Camera quality  
""") 

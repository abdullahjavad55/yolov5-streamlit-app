# app.py
import streamlit as st
import yolo_detector
from PIL import Image
import tempfile

st.set_page_config(page_title="YOLOv5 Object Detection", layout="centered")
st.title("🔍 YOLOv5 Object Detection App")

# Radio selection for image or video
option = st.radio("Choose input type:", ("Image", "Video"))

if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            result_image = yolo_detector.detect_image(tmp.name)
            st.image(result_image, caption="Detected Objects", use_container_width=True)

elif option == "Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        st.video(tfile.name)
        st.text("🔄 Running detection... please wait.")
        output_path = yolo_detector.detect_video(tfile.name)

        st.success("✅ Detection complete.")
        st.video(output_path)

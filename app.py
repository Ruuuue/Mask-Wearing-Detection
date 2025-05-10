import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# --- Page Config ---
st.set_page_config(
    page_title="Mask Wearing Detection",
    page_icon="😷",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Load YOLO Model ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# --- Header ---
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>😷 Mask Wearing Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image to check for mask compliance using YOLOv8</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Image Upload ---
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Detecting... Please wait"):
        img_array = np.array(image)
        results = model(img_array)
        result_img = results[0].plot()

    st.markdown("---")
    st.subheader("🧾 Detection Result")
    st.image(result_img, caption="Processed Image", use_column_width=True)
    st.success("✅ Detection completed!")
else:
    st.info("📤 Upload an image to get started.")

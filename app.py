import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# --- Streamlit page config ---
st.set_page_config(page_title="😷 Mask Detection", layout="centered")

# --- Load YOLO model ---
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    # Manually define class names if needed
    model.names = {
        0: "with_mask",
        1: "without_mask",
        2: "mask_weared_incorrect"
    }
    return model

model = load_model()

# --- App Header ---
st.title("😷 Mask Wearing Detection")
st.write("Upload an image to detect mask compliance using a YOLOv8 model trained on Roboflow.")

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Detecting... Please wait"):
        img_array = np.array(image)
        results = model.predict(source=img_array, conf=0.5)

        annotated_image = results[0].plot()

        # Show detection image
        st.subheader("🔍 Detection Output")
        st.image(annotated_image, caption="Detection Result", use_column_width=True)

        # Count detections
        counts = {label: 0 for label in model.names.values()}
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            counts[label] += 1
            st.write(f"• **{label}** — {conf:.2f}")

        # Summary
        st.subheader("📊 Summary:")
        for label, count in counts.items():
            st.write(f"- `{label}`: {count} detected")

else:
    st.info("📤 Upload an image to get started.")

import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# --- Configuration ---
MODEL_PATH = "best.pt"
INFERENCE_IMAGE_SIZE = 640  # Use None to let the model decide

# OPTIONAL: Define custom class names manually if needed
custom_class_names = {
    0: "with_mask",
    1: "without_mask",
    2: "mask_weared_incorrect"
}

st.title("😷 Mask Wearing Detection")
st.write(f"Upload an image to see detections from your '{os.path.basename(MODEL_PATH)}' model.")

# --- Load Model ---
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")
    st.stop()

# --- File Upload ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image_pil = Image.open(uploaded_file).convert("RGB")
        st.image(image_pil, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Running detection..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img_file:
                image_pil.save(temp_img_file.name)
                temp_img_path = temp_img_file.name

            try:
                results = model.predict(temp_img_path, imgsz=INFERENCE_IMAGE_SIZE)

                result_img_array_bgr = results[0].plot()

                st.image(result_img_array_bgr, caption="Detected Image", channels="BGR", use_container_width=True)

                # Show detected classes and counts
                st.subheader("📋 Detected Classes:")
                counts = {label: 0 for label in custom_class_names.values()}
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    label = custom_class_names.get(cls_id, f"class_{cls_id}")
                    conf = float(box.conf[0])
                    counts[label] += 1
                    st.write(f"- **{label}** — {conf:.2f}")

                # Summary
                st.subheader("📊 Summary:")
                for label, count in counts.items():
                    st.write(f"- `{label}`: {count} detected")

            except Exception as e:
                st.error(f"Error during YOLO prediction: {e}")
            finally:
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)

    except Exception as e:
        st.error(f"Error processing image: {e}")
else:
    st.info("Please upload an image file to start detection.")

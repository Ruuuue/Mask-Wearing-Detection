import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# --- Configuration ---
# 1. Load your trained YOLOv11n model
#    Ensure "best.pt" is the correct path to your model file.
#    If "best.pt" is not in the same directory as your script, provide the full path.
MODEL_PATH = "best.pt"

# 2. (Optional) Specify image size for inference
#    The model will resize images to this size internally during prediction.
#    If None, the model will use its default or trained input size.
#    Common sizes are 320, 416, 640, etc.
INFERENCE_IMAGE_SIZE = 640 # Example: use 640, or set to None for model default

st.title("YOLOv11n Object Detection App")
st.write(f"Upload an image to see detections from your '{os.path.basename(MODEL_PATH)}' model.")

# --- Load Model ---
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")
    st.stop() # Stop execution if model can't be loaded

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load the uploaded image using PIL
        image_pil = Image.open(uploaded_file).convert("RGB")
        st.image(image_pil, caption="Uploaded Image", use_column_width=True)

        # --- Run YOLO Prediction ---
        with st.spinner("Running detection..."):
            # Save the PIL image to a temporary file, as YOLO model.predict() often works best with file paths
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img_file:
                image_pil.save(temp_img_file.name)
                temp_img_path = temp_img_file.name

            try:
                # Perform inference
                # The model internally resizes 'temp_img_path' to INFERENCE_IMAGE_SIZE (if specified)
                # or its default input size for processing.
                if INFERENCE_IMAGE_SIZE:
                    results = model.predict(temp_img_path, imgsz=INFERENCE_IMAGE_SIZE)
                else:
                    results = model.predict(temp_img_path)

                # The results[0].plot() method draws the detections on a copy of the original image
                # (or rather, an image of the same dimensions as the input to predict).
                # It returns a NumPy array (typically in BGR format).
                result_img_array_bgr = results[0].plot() # This is a NumPy array

                # Display the image with detections
                # st.image can handle NumPy arrays. Specify channels as "BGR" because
                # OpenCV (used by plot()) typically outputs BGR images.
                st.image(result_img_array_bgr, caption="Detected Image", channels="BGR", use_column_width=True)

            except Exception as e:
                st.error(f"Error during YOLO prediction: {e}")
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)

    except Exception as e:
        st.error(f"Error processing image: {e}")
else:
    st.info("Please upload an image file to start detection.")

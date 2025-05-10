# 😷 Mask Wearing Detection

A modern Streamlit app that uses a YOLOv8 model to detect whether people in an uploaded image are wearing masks or not.

---

## 📁 Project Overview

This project leverages the power of **YOLOv8** (You Only Look Once – version 8) to detect whether individuals in an uploaded image are wearing face masks or not. It's wrapped in a **modern, interactive Streamlit web app**, making it incredibly easy for anyone to try out the model without needing to touch a single line of code.

The goal is to provide a lightweight, fast, and accessible tool that can be used for:

- Public safety & compliance demonstrations
- AI model showcasing in web app form
- Real-time face mask detection in static images

The detection model was trained using images labeled as either `mask` or `no_mask`. The training process is documented inside the Jupyter notebook provided in this repo. It includes:

- Dataset loading and preprocessing (via Roboflow)
- YOLOv11 model configuration and training
- Evaluation and testing
- Saving the best-performing model (`best.pt`)

---

## 🔗 Live Demo

👉 [Click here to try the app](https://mask-wearing-detection-cpvwzzysmf2czzsebfamxu.streamlit.app/)

---

## 🛠️ How to Run Locally

1. **Clone the repo:**

```bash
git clone https://github.com/Ruuuue/mask-wearing-detection.git
cd mask-wearing-detection
```
2. **Install the required packages:**

```bash
pip install -r requirements.txt
```
3. **Add the model file:**
- Make sure best.pt is placed in the root folder of the project.

4. **Run the app:**

```bash
streamlit run app.py
```

import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠")

st.title("🧠 Brain Tumor Classification App")

# Load model once (important for performance)
@st.cache_resource
def load_trained_model():
    return load_model("resnet_finetuned.h5")

model = load_trained_model()

class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_container_width=True)

    # Preprocess
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = img.reshape(1, 224, 224, 3)

    with st.spinner("Analyzing MRI..."):
        prediction = model.predict(img)
    
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")

import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# ---------- Page config ----------
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide"
)

# ---------- CSS Styling ----------
st.markdown("""
<style>
.main-title {
    font-size:42px;
    font-weight:700;
    text-align:center;
    color:#4CAF50;
}

.card {
    padding:20px;
    border-radius:15px;
    background-color:#1f2937;
    box-shadow:0px 4px 20px rgba(0,0,0,0.25);
    margin-top:20px;
}

.result {
    font-size:28px;
    font-weight:bold;
    color:#4CAF50;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🧠 Brain Tumor Classification App</p>', unsafe_allow_html=True)

# ---------- Sidebar ----------
page = st.sidebar.selectbox("Navigation", ["Predict", "About Model"])

# ---------- Load model ----------
@st.cache_resource
def load_trained_model():
    return load_model("resnet_finetuned.h5")

model = load_trained_model()

class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# ---------- Predict Page ----------
if page == "Predict":

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        with col1:
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

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)

            st.markdown(f'<p class="result">Prediction: {predicted_class}</p>', unsafe_allow_html=True)
            st.metric("Confidence", f"{confidence:.2f}%")

            st.subheader("Class Probabilities")
            st.bar_chart(prediction[0])

            st.markdown('</div>', unsafe_allow_html=True)

# ---------- About Page ----------
elif page == "About Model":
    st.header("About Model")
    st.write("""
This app uses a **fine-tuned ResNet50 deep learning model** trained on MRI brain tumor images.

Classes:
- Glioma
- Meningioma
- Pituitary
- No Tumor

The model performs multiclass classification with transfer learning and data augmentation.
""")

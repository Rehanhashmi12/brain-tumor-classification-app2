import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# ---------- Page config ----------
st.set_page_config(
    page_title="Brain Tumor AI Detector",
    page_icon="🧠",
    layout="wide"
)

# ---------- PREMIUM GLASS CSS ----------
st.markdown("""
<style>

/* Tech gradient background */
.stApp{
background: radial-gradient(circle at top,#0f2027,#203a43,#2c5364);
}

/* HERO TITLE */
.big-title{
font-size:68px;
font-weight:900;
text-align:center;
color:white;
letter-spacing:1px;
text-shadow:0 0 25px rgba(0,255,255,0.6);
margin-bottom:10px;
}

.sub{
text-align:center;
color:#cbd5e1;
margin-bottom:40px;
font-size:18px;
}

/* GLASS CARD */
.card{
padding:28px;
border-radius:24px;
background:rgba(255,255,255,0.08);
backdrop-filter:blur(18px) saturate(180%);
-webkit-backdrop-filter:blur(18px) saturate(180%);
border:1px solid rgba(255,255,255,0.15);
box-shadow:0 10px 40px rgba(0,0,0,0.45);
margin-top:20px;
}

/* Prediction glow */
.result{
font-size:34px;
font-weight:700;
color:#00ffe5;
text-shadow:0 0 15px rgba(0,255,229,0.6);
}

/* Sidebar glass */
section[data-testid="stSidebar"]{
background:rgba(255,255,255,0.05);
backdrop-filter:blur(18px);
border-right:1px solid rgba(255,255,255,0.1);
}

</style>
""", unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown('<div class="big-title">🧠 Brain Tumor AI Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Upload MRI → AI predicts tumor type instantly</div>', unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Predict", "About Model"])

# ---------- Load model ----------
@st.cache_resource
def load_trained_model():
    return load_model("resnet_finetuned.h5")

model = load_trained_model()
class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# =========================
# PREDICT PAGE
# =========================
if page == "Predict":

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        with col1:
            st.image(image, use_container_width=True)

        # preprocess
        img = np.array(image)
        img = cv2.resize(img,(224,224))
        img = img/255.0
        img = img.reshape(1,224,224,3)

        with st.spinner("🧠 AI analyzing MRI..."):
            prediction = model.predict(img)

        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)*100

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)

            st.markdown(f'<div class="result">Prediction: {predicted_class}</div>', unsafe_allow_html=True)
            st.metric("Confidence", f"{confidence:.2f}%")

            st.subheader("Class Probabilities")
            st.progress(float(np.max(prediction)))
            st.bar_chart(prediction[0])

            st.markdown('</div>', unsafe_allow_html=True)

# =========================
# ABOUT PAGE
# =========================
elif page == "About Model":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.header("About Model")

    st.write("""
This application uses a **fine-tuned ResNet50 deep learning model** trained on brain MRI images to classify tumor types.

### Classes
• Glioma  
• Meningioma  
• Pituitary  
• No Tumor  

### Model Improvements
✔ Transfer learning  
✔ Fine-tuning  
✔ Data augmentation  
✔ Class imbalance handling  

### Impact
Demonstrates how AI can support radiologists in faster and more accurate brain tumor screening.
""")

    st.markdown('</div>', unsafe_allow_html=True)

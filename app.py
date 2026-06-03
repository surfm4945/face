import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

# 1. Page Configuration & Custom CSS Styling
st.set_page_config(
    page_title="SentientFace AI", 
    page_icon="🧠", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom branding banner via CSS injection
st.markdown("""
    <style>
    .main-title { font-size: 2.6rem; font-weight: 800; color: #00FF66; margin-bottom: 0.5rem; }
    .sub-title { font-size: 1.1rem; color: #9CA3AF; margin-bottom: 2rem; }
    .metric-card { background-color: #1F2937; padding: 1.5rem; border-radius: 0.75rem; border-left: 5px solid #00FF66; }
    </style>
""", unsafe_allow_html=True)

# 2. Sidebar Controls & Documentation
with st.sidebar:
    st.markdown("## 🧠 SentientFace Engine")
    st.markdown("An advanced Deep Learning inference platform analyzing facial micromovements in real-time.")
    st.divider()
    st.subheader("Model Meta-Data")
    st.caption("Architecture: Custom CNN (Convolutional Neural Network)")
    st.caption("Input Array Dimension: 48x48x1 (Grayscale)")
    st.caption("Target Variables: 7 Distinct Facial Expressions")
    st.divider()
    st.success("System Status: Operational")

# 3. Main Interface Layout
st.markdown('<div class="main-title">🧠 SentientFace AI Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Computer vision model processing biological indicators and expression analytics.</div>', unsafe_allow_html=True)

MODEL_PATH = 'mood_model.keras'
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Error Handling with elegant callouts
if not os.path.exists(MODEL_PATH):
    st.error(f"🚨 Operational Failure: Model binary matrix file missing at `{MODEL_PATH}`. Please upload your mood_model.keras file directly to the GitHub main folder.")
else:
    @st.cache_resource
    def load_engine():
        return tf.keras.models.load_model(MODEL_PATH)
    
    model = load_engine()

    # Split workspace into 2 balanced columns (Camera on Left | Analytics on Right)
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📷 Biometric Optical Stream")
        img_file = st.camera_input("Position face clearly in frame and initialize capture")

    with col2:
        st.markdown("### 📊 Live Analytics & Inference")
        
        if img_file is not None:
            # Decode image matrix array safely
            bytes_data = img_file.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Neural Net Pipeline Preprocessing
            gray_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
            resized_img = cv2.resize(gray_img, (48, 48))
            normalized_img = resized_img / 255.0
            reshaped_img = np.reshape(normalized_img, (1, 48, 48, 1))

            # Run Math Inference
            with st.spinner("Executing tensor matrix operations..."):
                predictions = model.predict(reshaped_img)
                max_idx = int(np.argmax(predictions))
                predicted_emotion = EMOTIONS[max_idx]
                confidence = predictions[0][max_idx] * 100

            # Beautiful High-Impact Metric Dashboard Card
            st.markdown(f"""
                <div class="metric-card">
                    <span style='color: #9CA3AF; text-transform: uppercase; font-size: 0.8rem; font-weight: 700;'>Primary Detected State</span>
                    <h2 style='margin: 0.2rem 0; color: #FFFFFF; font-size: 2.2rem;'>{predicted_emotion}</h2>
                    <span style='color: #00FF66; font-weight: 600;'>{confidence:.1f}% Model Certainty</span>
                </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            st.markdown("#### Complete Probability Distribution Vector")
            
            # Custom confidence tracker gauges
            for emo, prob in zip(EMOTIONS, predictions[0]):
                prob_pct = float(prob) * 100
                st.progress(float(prob), text=f"**{emo}** — {prob_pct:.1f}%")
        else:
            st.info("💡 Awaiting input stream. Activate the biometric optical camera on the left to execute predictions.")

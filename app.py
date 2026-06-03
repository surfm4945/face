import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import datetime
from tensorflow.keras.models import load_model

# --- Page Configurations ---
st.set_page_config(page_title="AI Mood Analyzer", layout="wide")
st.title("🧠 Local Real-time Mood Analyzer")
st.markdown("Analyze your facial expressions instantly using your webcam and local deep learning.")

# --- Constants & Setup ---
MOOD_CLASSES = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
CSV_FILE = 'mood_history.csv'

# Initialize CSV file if it doesn't exist
if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=['Timestamp', 'Mood', 'Confidence'])
    df.to_csv(CSV_FILE, index=False)

# Load OpenCV Face Detector Cascade
# Download link for file: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
cascade_path = "haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_path):
    st.error(f"Missing '{cascade_path}'. Please place it in the root project folder.")
    st.stop()
face_cascade = cv2.CascadeClassifier(cascade_path)

# Load the Trained CNN Model
@st.cache_resource
def load_local_model():
    try:
        return load_model('models/mood_model.keras')
    except Exception as e:
        st.error("Could not find 'models/mood_model.keras'. Please execute train.py first.")
        st.stop()

model = load_local_model()

# --- Helper Functions ---
def log_mood(mood, confidence):
    """Logs the detected mood to a local CSV file."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = pd.DataFrame([[now, mood, confidence]], columns=['Timestamp', 'Mood', 'Confidence'])
    new_entry.to_csv(CSV_FILE, mode='a', header=False, index=False)

def get_daily_trend(df):
    """Calculates overall context trend using the simple majority of entries."""
    if df.empty:
        return "No Data"
    top_mood = df['Mood'].mode()[0]
    if top_mood in ['Happy', 'Surprise']:
        return "Positive ☀️"
    elif top_mood in ['Sad', 'Angry']:
        return "Negative 🌧️"
    else:
        return "Neutral ☁️"

# --- Layout Grid Separation ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Camera Feed")
    run_cam = st.checkbox("Toggle Web Camera Switch", value=False)
    frame_placeholder = st.empty()

with col2:
    st.subheader("📊 Session Diagnostics")
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()

# --- Core Inference Engine Loop ---
if run_cam:
    cap = cv2.VideoCapture(0)  # ID 0 defaults to the main laptop/USB webcam
    
    if not cap.isOpened():
        st.error("Error: Could not access your local webcam hardware.")
        run_cam = False

    while run_cam:
        ret, frame = cap.read()
        if not ret:
            st.warning("Skipping broken video stream frame inputs.")
            continue

        # Convert frame color domains for model compatibility
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Multi-scale face tracking bounding evaluations
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        current_mood = "No Face Detected"
        confidence_pct = 0.0

        for (x, y, w, h) in faces:
            # Draw rectangle around face boundary
            cv2.rectangle(rgb_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Isolate and preprocess region of interest (ROI)
            roi_gray = gray_frame[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            roi_normalized = roi_gray.astype('float32') / 255.0
            roi_batch = np.expand_dims(np.expand_dims(roi_normalized, -1), 0)

            # Execution inference passing grayscale image shape [1, 48, 48, 1]
            predictions = model.predict(roi_batch, verbose=0)[0]
            max_index = np.argmax(predictions)
            
            current_mood = MOOD_CLASSES[max_index]
            confidence_pct = float(predictions[max_index] * 100)

            # Render overlay label details directly onto visual pipeline output
            label = f"{current_mood} ({confidence_pct:.1f}%)"
            cv2.putText(rgb_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Log periodic snapshots (prevents flooding the storage array too fast)
            if np.random.rand() < 0.15:
                log_mood(current_mood, confidence_pct)

        # Live Display Update
        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

        # Dynamic Metrics Updates
        try:
            history_df = pd.read_csv(CSV_FILE)
            trend = get_daily_trend(history_df)
        except Exception:
            history_df = pd.DataFrame()
            trend = "No Data"

        with metrics_placeholder.container():
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Current State Evaluation", current_mood)
            m_col2.metric("Trend Track Assessment", trend)
            if confidence_pct > 0:
                st.progress(confidence_pct / 100.0, text=f"Confidence Accuracy Strength: {confidence_pct:.1f}%")

        with chart_placeholder.container():
            if not history_df.empty:
                st.write("**Mood Log History over Time:**")
                # Group data points to display timeline metrics easily
                chart_data = history_df.tail(20)
                st.line_chart(chart_data.set_index('Timestamp')['Confidence'])

    cap.release()
else:
    st.info("Check the checkbox component toggle above to boot internal webcam drivers.")
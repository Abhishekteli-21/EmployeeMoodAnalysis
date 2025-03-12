import streamlit as st
import cv2
import numpy as np
import speech_recognition as sr
from src.emotion_detector import EmotionDetector
from src.database import MoodDatabase
from src.analytics import MoodAnalytics
from src.utils import load_config
import pandas as pd

# Load configuration
config = load_config()
detector = EmotionDetector()
db = MoodDatabase()
analytics = MoodAnalytics()

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f4f7fa;}
    .stButton>button {background-color: #28a745; color: white; border-radius: 8px; padding: 10px;}
    .stTextInput>input, .stTextArea>textarea {border: 2px solid #007bff; border-radius: 8px;}
    .stSelectbox {background-color: #ffffff; border: 2px solid #007bff; border-radius: 8px;}
    .header {color: #007bff; font-size: 32px; font-weight: bold; text-align: center;}
    .subheader {color: #444444; font-size: 22px; font-weight: bold;}
    .badge {background-color: #ff9800; color: white; padding: 6px 12px; border-radius: 20px; font-size: 14px;}
    .sidebar .sidebar-content {background-color: #e9ecef;}
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h2 style='color: #007bff;'>Settings</h2>", unsafe_allow_html=True)
    employee_id = st.text_input("Employee ID", "EMP001", help="Your unique identifier")
    mic_options = sr.Microphone.list_microphone_names()
    mic_index = st.selectbox("Microphone", range(len(mic_options)), format_func=lambda x: mic_options[x], help="Choose your mic")
    st.markdown("---")
    st.info("Control your emotion analysis session.")

# Main Interface
st.markdown(f"<div class='header'>{config['app']['title']}</div>", unsafe_allow_html=True)
tab1, tab2 = st.tabs(["Real-Time Detection", "Analytics"])

# Tab 1: Real-Time Detection
with tab1:
    st.markdown("<div class='subheader'>Live Emotion Tracking</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### Video Stream", unsafe_allow_html=True)
        video_placeholder = st.empty()
        video_status = st.empty()
        start_button = st.button("▶️ Start Video Analysis", help="Begin video emotion detection")
        stop_button = st.button("⏹️ Stop Video Analysis", help="End video analysis and process results")

        if "analyzing" not in st.session_state:
            st.session_state["analyzing"] = False
        if "video_emotion" not in st.session_state:
            st.session_state["video_emotion"] = "neutral"

        if start_button:
            st.session_state["analyzing"] = True
            video_status.write("Analyzing video...")

            def update_frame(frame, emotion):
                cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                video_placeholder.image(frame, channels="BGR")

            video_emotion, emotions = detector.analyze_video_duration(config["emotion_detection"]["video_duration"], update_frame)
            st.session_state["video_emotion"] = video_emotion if emotions else "neutral"
            video_status.success(f"Video Analysis Complete: Dominant Emotion - {st.session_state['video_emotion']}")

        if stop_button and st.session_state["analyzing"]:
            st.session_state["analyzing"] = False
            video_status.success(f"Video Analysis Stopped: Dominant Emotion - {st.session_state['video_emotion']}")

        if not st.session_state["analyzing"]:
            video_placeholder.image(np.zeros((480, 640, 3), dtype=np.uint8), channels="BGR", caption="Video Stopped")

    with col2:
        st.markdown("#### Input Analysis", unsafe_allow_html=True)
        text_input = st.text_area("Text Input (Optional)", placeholder="How are you feeling today?", height=120)
        text_emotion = "neutral"
        if st.button("Analyze Text", help="Process text input"):
            text_emotion = detector.detect_text_emotion(text_input)
        st.markdown(f"<span class='badge'>Text: {text_emotion}</span>", unsafe_allow_html=True)

        speech_status = st.empty()
        if st.button("🎤 Record Speech", help="Record your voice"):
            with st.spinner("Listening..."):
                speech_emotion = detector.detect_speech_emotion(device_index=mic_index)
                speech_status.markdown(f"<span class='badge'>Speech: {speech_emotion}</span>", unsafe_allow_html=True)
        else:
            speech_emotion = "neutral"
            speech_status.markdown("<span class='badge'>Speech: neutral</span>", unsafe_allow_html=True)

        stress_score = detector.compute_stress_score(st.session_state["video_emotion"], text_emotion, speech_emotion)
        st.progress(stress_score)
        st.markdown(f"Stress Score: **{stress_score:.2f}**")

        if st.button("Save Analysis", help="Store current results"):
            db.insert_mood(employee_id, st.session_state["video_emotion"], text_emotion, speech_emotion, stress_score)
            st.success("Results saved to database!")
            if analytics.check_stress_alert(employee_id):
                st.warning("⚠️ Prolonged Stress Detected - HR Notified via Email")

# Tab 2: Analytics
with tab2:
    st.markdown("<div class='subheader'>Team & Individual Insights</div>", unsafe_allow_html=True)
    task = analytics.get_task_recommendation(employee_id)
    st.success(f"Recommended Task: **{task}**")

    historical = db.get_historical_data(employee_id)
    if historical:
        df = pd.DataFrame(historical)
        st.markdown("#### Historical Mood Tracking", unsafe_allow_html=True)
        st.dataframe(df[["timestamp", "video_emotion", "text_emotion", "speech_emotion", "stress_score"]], use_container_width=True)
        trend_fig = analytics.plot_mood_trend(employee_id)
        if trend_fig:
            st.plotly_chart(trend_fig, use_container_width=True)

    team_data = analytics.team_mood_summary()
    if not team_data.empty:
        st.markdown("#### Team Mood Analytics", unsafe_allow_html=True)
        st.dataframe(team_data, use_container_width=True)
        heatmap_fig = analytics.plot_team_heatmap()
        if heatmap_fig:
            st.plotly_chart(heatmap_fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666666;'></p>", unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        st.session_state["running"] = True
    finally:
        db.close()
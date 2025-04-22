import sys
import os

# Add path to local stress_analysis_refactored folder
sys.path.append(os.path.join(os.path.dirname(__file__), 'stress_analysis_refactored'))

import streamlit as st
import matplotlib.pyplot as plt
import tempfile
from tensorflow.keras.models import load_model  # type: ignore
from Function import get_video_duration, process_emotions, process_eye_gaze

# Load Models
emotion_model = load_model('stress_analysis/Model/Emotions_Model.h5')
eye_gaze_model = load_model('stress_analysis/Model/EyeGaze_Model.h5')

# Set the page title
st.title("Facial Expression and Eye Gaze Detection")

# Video upload functionality
uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"], key="expression_video")

if uploaded_video is not None:
    st.video(uploaded_video)
    st.success("Video uploaded successfully!")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(uploaded_video.read())
        temp_video_path = temp_file.name

    video_duration = get_video_duration(temp_video_path)

    # --- Facial Expression Analysis ---
    st.header("🎭 Facial Expression Analysis")

    try:
        emotion_counts, total_frames, max_emotion, average_emotion_score = process_emotions(
            temp_video_path, emotion_model, video_duration
        )

        if not emotion_counts.empty:
            fig, ax = plt.subplots()
            ax.bar(emotion_counts['Emotions'], emotion_counts['Frames'],
                   color=["#FF9999", "#66B2FF", "#99FF99", "#FFCC99", "#FFD700", "#87CEFA", "#90EE90"])
            ax.set_xlabel('Emotions')
            ax.set_ylabel('Frames')
            ax.set_title('Emotion Distribution')
            ax.set_xticks(range(len(emotion_counts['Emotions'])))
            ax.set_xticklabels(emotion_counts['Emotions'], rotation=45)
            st.pyplot(fig)

            st.write("### Emotion Distribution Table")
            st.table(emotion_counts)
            st.write(f"#### Total Frames: {total_frames}")
            st.write(f"### Predominant Emotion: {max_emotion}")
            st.success(f"🎯 Facial Emotion Score: {average_emotion_score:.2f} / 10")
        else:
            raise ValueError("Empty emotion count.")

    except Exception as e:
        average_emotion_score = 0
        st.error(f"⚠️ Facial Expression Analysis failed: {e}")
        st.info(f"🎯 Facial Emotion Score: {average_emotion_score:.2f} / 10")

    # --- Eye Gaze Analysis ---
    st.header("👀 Eye Gaze Analysis")

    try:
        eye_gaze_counts, total_eye_frames, max_eye_gaze, final_eye_gaze_score = process_eye_gaze(
            temp_video_path, eye_gaze_model, video_duration
        )

        if not eye_gaze_counts.empty:
            fig, ax = plt.subplots()
            ax.bar(eye_gaze_counts['Eye Gaze'], eye_gaze_counts['Frames'],
                   color=["#FF9999", "#66B2FF", "#99FF99"])
            ax.set_xlabel('Eye Gaze')
            ax.set_ylabel('Frames')
            ax.set_title('Eye Gaze Distribution')
            ax.set_xticks(range(len(eye_gaze_counts['Eye Gaze'])))
            ax.set_xticklabels(eye_gaze_counts['Eye Gaze'], rotation=45)
            st.pyplot(fig)

            st.write("### Eye Gaze Distribution Table")
            st.table(eye_gaze_counts)
            st.write(f"#### Total Frames: {total_eye_frames}")
            st.write(f"### Predominant Eye Gaze: {max_eye_gaze}")
            st.success(f"🎯 Eye Gaze Score: {final_eye_gaze_score:.2f} / 10")
        else:
            raise ValueError("Empty eye gaze count.")

    except Exception as e:
        final_eye_gaze_score = 0
        st.error(f"⚠️ Eye Gaze Analysis failed: {e}")
        st.info(f"🎯 Eye Gaze Score: {final_eye_gaze_score:.2f} / 10")
        
# --- Final Combined Score ---
st.header("📊 Final Combined Score")

combined_score = (average_emotion_score + final_eye_gaze_score) / 2
st.markdown(f"""
<div style='
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 28px;
    color: #2c3e50;
    font-weight: bold;
    border: 2px solid #4CAF50;
'>
    🏁 Final Score (Facial + Eye Gaze): <span style='color:#007ACC'>{combined_score:.2f} / 10</span>
</div>
""", unsafe_allow_html=True)
import streamlit as st
import matplotlib.pyplot as plt
import tempfile
from tensorflow.keras.models import load_model # type: ignore
from Function import get_video_duration, process_emotions, process_eye_gaze

# Load Models
emotion_model = load_model('Model/Emotions_Model.h5')
eye_gaze_model = load_model('Model/EyeGaze_Model.h5')

video_int = False
# Set the page title
st.title("Facial Expression and Eye Gaze Detection")

# Video upload functionality
uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"], key="expression_video")

if uploaded_video is not None:
    st.video(uploaded_video)
    st.success("Video uploaded successfully!")
    
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(uploaded_video.read())
        temp_video_path = temp_file.name

        # Get the video duration
        video_duration = get_video_duration(temp_video_path)

        video_int = True

# Create tabs
tab1, tab2 = st.tabs(["Facial Expression Detection", "Eye Gaze Detection"])

with tab1:
    if not video_int:
        st.write("Upload a video to view the emotion distribution.")
    else:
        # Process the uploaded video for facial expressions
        emotion_counts, total_frames, max_emotion, average_emotion_score = process_emotions(
            temp_video_path, emotion_model, video_duration
        )

        st.header("Facial Expression Distribution")
        if uploaded_video is not None and not emotion_counts.empty:
            # Create a bar chart based on the emotion counts
            fig, ax = plt.subplots()
            ax.bar(emotion_counts['Emotions'], emotion_counts['Frames'], color=["#FF9999", "#66B2FF", "#99FF99", "#FFCC99", "#FFD700", "#87CEFA", "#90EE90"])
            ax.set_xlabel('Emotions')
            ax.set_ylabel('Frames')
            ax.set_title('Emotion Distribution')
            ax.set_xticks(range(len(emotion_counts['Emotions'])))
            ax.set_xticklabels(emotion_counts['Emotions'], rotation=45)

            # Display the bar chart
            st.pyplot(fig)
            st.write("### Emotion Distribution")
            
            # Display the updated table
            st.table(emotion_counts)

            # Display the total frames
            st.write(f"#### Total Frames: {total_frames}")
            
            st.write(f"### The predominant facial expression of the candidate is {max_emotion} in this video")
            st.write(f"### Emotion Score (emotions > 0.5 seconds): {average_emotion_score:.2f}/10")

with tab2:
    if not video_int:
        st.write("Upload a video to view the eye gaze distribution.")
    else:
        # Process the uploaded video for eye gaze
        eye_gaze_counts, total_eye_frames, max_eye_gaze, final_eye_gaze_score = process_eye_gaze(
            temp_video_path, eye_gaze_model, video_duration
        )

        st.header("Eye Gaze Distribution")
        if uploaded_video is not None and not eye_gaze_counts.empty:
            # Create a bar chart based on the eye gaze counts
            fig, ax = plt.subplots()
            ax.bar(eye_gaze_counts['Eye Gaze'], eye_gaze_counts['Frames'], color=["#FF9999", "#66B2FF", "#99FF99"])
            ax.set_xlabel('Eye Gaze')
            ax.set_ylabel('Frames')
            ax.set_title('Eye Gaze Distribution')
            ax.set_xticks(range(len(eye_gaze_counts['Eye Gaze'])))
            ax.set_xticklabels(eye_gaze_counts['Eye Gaze'], rotation=45)

            # Display the bar chart
            st.pyplot(fig)
            st.write("### Eye Gaze Distribution")
            
            # Display the updated table
            st.table(eye_gaze_counts)

            # Display the total frames
            st.write(f"#### Total Frames: {total_eye_frames}")
            st.write(f"### The predominant eye gaze of the candidate is {max_eye_gaze} in this video")
            st.write(f"### Eye Gaze Score: {final_eye_gaze_score}/10")
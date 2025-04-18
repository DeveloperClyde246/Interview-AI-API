import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from Preprocessor import Preprocessor
import tempfile
import numpy as np
from tensorflow.keras.models import load_model

# Load Models
emotion_model = load_model('Model/Emotions_Model.h5')
eye_gaze_model = load_model('Model/EyeGaze_Model.h5')

# Define scores for each emotion
emotion_scores = {
    'Happy': 10,
    'Neutral': 8,
    'Surprise': 7,
    'Sad': 5,
    'Angry': 4,
    'Disgust': 2,
    'Fear': 1
}

# Define scores for each eye gaze type
eye_gaze_scores = {
    'Forward Look': 10,
    'Side Look (Left + Right)': 6,
    'Right Look': 6,
    'Left Look ': 6,
    'Close Look': 3
}

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

        # Assuming you have the video duration in seconds
        video_duration = Preprocessor.get_video_duration(temp_video_path)

        video_int = True

# Create tabs
tab1, tab2 = st.tabs(["Facial Expression Detection", "Eye Gaze Detection"])

with tab1:
    if video_int == False:
        st.write("Upload a video to view the emotion distribution.")
    else:
        # Process the uploaded video using Preprocessor
        preprocessor = Preprocessor()
        try:
            preprocessed_data = preprocessor.preprocessFER(temp_video_path)
            processed_frames = np.array(preprocessed_data)
            predictions = emotion_model.predict(processed_frames)
            predicted_emotions = np.argmax(predictions, axis=1)
        except Exception as e:
            st.warning(f"⚠️ Facial expression model failed: {e}")
            predicted_emotions = []
            average_emotion_score = 0

        if uploaded_video is not None and len(predicted_emotions) > 0:
            # Map predictions to emotion labels
            emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            emotion_counts = pd.Series(predicted_emotions).value_counts().sort_index()
            emotion_counts.index = [emotion_labels[i] for i in emotion_counts.index]

            st.header("Facial Expression Distribution")
            if uploaded_video is not None and len(predicted_emotions) > 0:
                # Create a bar chart based on the emotion counts
                fig, ax = plt.subplots()
                ax.bar(emotion_counts.index, emotion_counts.values, color=["#FF9999", "#66B2FF", "#99FF99", "#FFCC99", "#FFD700", "#87CEFA", "#90EE90"])
                ax.set_xlabel('Emotions')
                ax.set_ylabel('Frames')
                ax.set_title('Emotion Distribution')
                ax.set_xticks(range(len(emotion_counts.index)))
                ax.set_xticklabels(emotion_counts.index, rotation=45)

                # Display the bar chart
                st.pyplot(fig)
                st.write("### Emotion Distribution")
                
                # Rename the dataframe columns
                emotion_counts = emotion_counts.reset_index()
                emotion_counts.columns = ['Emotions', 'Frames']

                # Calculate the total frames
                total_frames = emotion_counts['Frames'].sum()

                # Calculate the percentage of frames for each emotion
                emotion_counts['Percentage'] = (emotion_counts['Frames'] / total_frames) * 100

                # Calculate the duration for each emotion
                emotion_counts['Duration (seconds)'] = (emotion_counts['Percentage'] / 100) * video_duration

                # Sort the DataFrame by 'Duration (seconds)' in descending order
                emotion_counts = emotion_counts.sort_values(by='Duration (seconds)', ascending=False)

                # Display the updated table
                st.table(emotion_counts)
                # Calculate the total frames

                total_frames = emotion_counts['Frames'].sum()

                # Display the total frames
                st.write(f"#### Total Frames: {total_frames}")
                
                # Get the maximum emotion
                max_emotion = emotion_counts.loc[emotion_counts['Frames'].idxmax()]['Emotions']

                # Check if the max emotion is "Neutral"
                if max_emotion == "Neutral":
                    # Filter emotions with duration greater than 0.5 seconds
                    filtered_emotions = emotion_counts[emotion_counts['Duration (seconds)'] > 0.5]
                    
                    # Exclude "Neutral" from the filtered emotions
                    filtered_emotions = filtered_emotions[filtered_emotions['Emotions'] != "Neutral"]
                    
                    # Check if there are any remaining emotions
                    if not filtered_emotions.empty:
                        # Get the emotion with the maximum duration from the filtered emotions
                        max_emotion = filtered_emotions.loc[filtered_emotions['Duration (seconds)'].idxmax()]['Emotions']

                # Filter emotions with a duration greater than 0.5 seconds
                filtered_emotions = emotion_counts[emotion_counts['Duration (seconds)'] > 0.5]

                # Calculate the average score for filtered emotions
                if not filtered_emotions.empty:
                    filtered_emotions['Score'] = filtered_emotions['Emotions'].map(emotion_scores)
                    average_emotion_score = filtered_emotions['Score'].mean()
                else:
                    average_emotion_score = 0  # Default to 0 if no emotions meet the criteria


                st.write(f"### The predominant facial expression of the candidate is {max_emotion} in this video")
                # Display the score
                st.write(f"### Emotion Score (emotions > 0.5 seconds): {average_emotion_score:.2f}/10")
            else:
                st.write("No valid predictions — Emotion Score: 0/10")



with tab2:
    if video_int == False:
        st.write("Upload a video to view the eye gaze distribution.")
    else:
        # Process the uploaded video using Preprocessor
        preprocessor = Preprocessor()
        preprocessed_eye_data = preprocessor.preprocessEyeGaze(temp_video_path)

        # Predict eye gaze for each frame
        processed_eye_frames = np.array(preprocessed_eye_data)
        try:
            eye_predictions = eye_gaze_model.predict(processed_eye_frames)
            predicted_eye_gaze = np.argmax(eye_predictions, axis=1)
        except Exception as e:
            st.warning(f"⚠️ Eye Gaze model failed: {e}")
            predicted_eye_gaze = []
            final_eye_gaze_score = 0

        if uploaded_video is not None and len(predicted_eye_gaze) > 0:
            # Map predictions to eye gaze labels
            eye_gaze_labels = ['Close Look', 'Forward Look', 'Left Look', 'Right Look']
            eye_gaze_counts = pd.Series(predicted_eye_gaze).value_counts().sort_index()
            eye_gaze_counts.index = [eye_gaze_labels[i] for i in eye_gaze_counts.index]

            st.header("Eye Gaze Distribution")
            if uploaded_video is not None and len(predicted_eye_gaze) > 0:
                # Create a bar chart based on the eye gaze counts
                fig, ax = plt.subplots()
                ax.bar(eye_gaze_counts.index, eye_gaze_counts.values, color=["#FF9999", "#66B2FF", "#99FF99"])
                ax.set_xlabel('Eye Gaze')
                ax.set_ylabel('Frames')
                ax.set_title('Eye Gaze Distribution')
                ax.set_xticks(range(len(eye_gaze_counts.index)))
                ax.set_xticklabels(eye_gaze_counts.index, rotation=45)

                # Display the bar chart
                st.pyplot(fig)
                st.write("### Eye Gaze Distribution")
                
                # Rename the dataframe columns
                eye_gaze_counts = eye_gaze_counts.reset_index()
                eye_gaze_counts.columns = ['Eye Gaze', 'Frames']

                # Combine "Left Look" and "Right Look" into "Side Look"
                if 'Left Look' in eye_gaze_counts['Eye Gaze'].values and 'Right Look' in eye_gaze_counts['Eye Gaze'].values:
                    side_look_frames = eye_gaze_counts.loc[eye_gaze_counts['Eye Gaze'] == 'Left Look', 'Frames'].values[0] + \
                                    eye_gaze_counts.loc[eye_gaze_counts['Eye Gaze'] == 'Right Look', 'Frames'].values[0]
                    # Remove "Left Look" and "Right Look"
                    eye_gaze_counts = eye_gaze_counts[~eye_gaze_counts['Eye Gaze'].isin(['Left Look', 'Right Look'])]
                    # Add "Side Look"
                    eye_gaze_counts = pd.concat([eye_gaze_counts, pd.DataFrame({'Eye Gaze': ['Side Look (Left + Right)'], 'Frames': [side_look_frames]})], ignore_index=True)

                # Calculate the total frames
                total_eye_frames = eye_gaze_counts['Frames'].sum()

                # Calculate the percentage of frames for each eye gaze
                eye_gaze_counts['Percentage'] = (eye_gaze_counts['Frames'] / total_eye_frames) * 100

                # Calculate the duration for each eye gaze
                eye_gaze_counts['Duration (seconds)'] = (eye_gaze_counts['Percentage'] / 100) * video_duration

                # Sort the DataFrame by 'Duration (seconds)' in descending order
                eye_gaze_counts = eye_gaze_counts.sort_values(by='Duration (seconds)', ascending=False)

                st.table(eye_gaze_counts)

                # Display the total frames
                st.write(f"#### Total Frames: {total_eye_frames}")
                # Display the message for the maximum eye gaze
                max_eye_gaze = eye_gaze_counts.loc[eye_gaze_counts['Frames'].idxmax()]['Eye Gaze']

                # Get the score for the predominant eye gaze
                final_eye_gaze_score = eye_gaze_scores.get(max_eye_gaze, 0)

                st.write(f"### The predominant eye gaze of the candidate is {max_eye_gaze} in this video")# Display the score
                st.write(f"### Eye Gaze Score: {final_eye_gaze_score}/10")
        else:
            st.write("No valid predictions — Eye Gaze Score: 0/10")



import streamlit as st
import os
import sys
import langcodes

# ✅ Add this to allow imports from speech-score folder
sys.path.append(os.path.join(os.path.dirname(__file__), 'speech_score'))

from function_class import (
    convert_video_to_audio, transcribe_audio, preprocess_text,
    translate_to_indonesian, load_model_and_resources,
    predict_stress, predict_emotion
)

# RUN: python -m streamlit run "C:\Users\Admin\OneDrive\Desktop\speech-score\main-score.py"

st.title('Candidate Text-Based Scoring System')

# Accept video file input
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])

# Process the uploaded video
if uploaded_video is not None:
    st.video(uploaded_video)
    st.success("Video uploaded successfully!")

    # Single spinner for the entire process
    with st.spinner("Processing your request, please wait..."):
        progress_bar = st.progress(0)  # Initialize progress bar
        
        # Convert video to audio (10%)
        progress_bar.progress(10)
        audio_file = convert_video_to_audio(uploaded_video)

        if audio_file:
            # Transcribe audio (30%)
            progress_bar.progress(30)
            transcription_result = transcribe_audio(audio_file)
            speech_text = transcription_result["text"]
            detected_language = transcription_result["language"]

            st.write("**Detected Language:**", langcodes.get(detected_language).display_name())
            st.write("**Transcribed Text:**", speech_text)

            # Translate to Indonesian if necessary (50%)
            if detected_language != "id":
                speech_text = translate_to_indonesian(speech_text)
                st.write("**Translated Text:**", speech_text)

            # Preprocess text (70%)
            progress_bar.progress(70)
            original_text = speech_text
            speech_text = preprocess_text(speech_text)

            # Clean up audio file (80%)
            os.remove(audio_file)

            # Stress Model 
            # Load model and predict (90%)
            progress_bar.progress(90)
            model, tokenizer, label_encoder = load_model_and_resources()
            stress_output = predict_stress(speech_text, model, tokenizer, label_encoder)
            
            # Show Output
            progress_bar.progress(100)  # Set progress to 100%
            # st.write("Predicted Output:", output)

            # Display Stress Detection Result
            st.header("Presence of Stress")
            if stress_output == 1:  # Assuming output == 1 means stress detected
                st.markdown(
                    """
                    <div style="background-color: #FFDDC1; padding: 15px; border-radius: 10px; text-align: center;">
                        <h3 style="color: #E63946;">Stress Detected</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div style="background-color: #C6F6D5; padding: 15px; border-radius: 10px; text-align: center;">
                        <h3 style="color: #2D6A4F;">No Stress detected</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            # Emotion Model
            # Emotion Analysis Section
            st.header("Emotion Analysis")
            
            with st.spinner("Analyzing emotion..."):
                predicted_emotion = predict_emotion(original_text)
            
            # Display result with emoji
            emotion_emojis = {
                'anger': '😠',
                'fear': '😨',
                'love': '🥰',
                'sadness': '😢',
                'happy': '😄',
                'Unknown': '❓'
            }
            
            emoji = emotion_emojis.get(predicted_emotion, '❓')
            st.success(f"Predicted emotion: {predicted_emotion.capitalize()} {emoji}")
                
        else:
            st.error("Failed to extract audio from the video.")
        
        # Display the scoring table. 
        # Streamlit app
        st.title("Emotion-Stress Impact Analysis")

        # HTML table with custom styling
        st.markdown("""
        <style>
            .data-table {
                width: 100%;
                border-collapse: collapse;
                margin: 25px 0;
                font-size: 0.9em;
                font-family: sans-serif;
                box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
            }
            .data-table thead tr {
                background-color: #009879;
                color: #ffffff;
                text-align: left;
            }
            .data-table th,
            .data-table td {
                padding: 12px 15px;
                border: 1px solid #dddddd;
            }
            .data-table tbody tr {
                border-bottom: 1px solid #dddddd;
            }
            .data-table tbody tr:nth-of-type(even) {
                background-color: #f3f3f3;
            }
            .data-table tbody tr:last-of-type {
                border-bottom: 2px solid #009879;
            }
            .bullet-points {
                margin-left: 20px;
                padding-left: 5px;
            }
        </style>

        <table class="data-table">
            <thead>
                <tr>
                    <th>Stress</th>
                    <th>Emotions</th>
                    <th>Score</th>
                    <th>Statement</th>
                </tr>
            </thead>
            <tbody>
                    <tr>
                    <td>No</td>
                    <td>Joy/Love</td>
                    <td>4</td>
                    <td class="bullet-points">
                        Optimal performance
                    </td>
                </tr>
                    <tr> 
                    <td>Yes</td>
                    <td>Happy/Love</td>
                    <td>3</td>
                    <td class="bullet-points">
                        Resilient but strained
                    </td>
                </tr>
                    <tr>
                    <td>No</td>
                    <td>Anger/Fear/Sadness</td>
                    <td>2</td>
                    <td class="bullet-points">
                        Temporary frustration (monitor, no immediate action)
                    </td>
                </tr>
                <tr>
                    <td>Yes</td>
                    <td>Anger/Fear/Sadness</td>
                    <td>1</td>
                    <td class="bullet-points">
                        High burnout risk (act urgently)
                    </td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)

        # Define emotion mappings
        NEGATIVE_EMOTIONS = ['anger', 'fear', 'sadness']  # Using Indonesian terms from your model
        POSITIVE_EMOTIONS = ['happy', 'love']  # Add other positive emotions as needed

        # Calculate combined score
        if stress_output == 1:
            if predicted_emotion.lower() in NEGATIVE_EMOTIONS:
                combine_score = 1
            else:
                combine_score = 3
        elif stress_output == 0:
            if predicted_emotion.lower() in NEGATIVE_EMOTIONS:
                combine_score = 2
            else:
                combine_score = 4
        else:
            combine_score = 0  # Unknown case

        # Display the result in Streamlit
        st.markdown(f"""
                    
                    <div style='
                        background-color: #f0f2f6;
                        padding: 15px;
                        border-radius: 10px;
                        text-align: center;
                        font-size: 24px;
                        color: #2c3e50;
                        font-weight: bold;
                        border: 2px solid #4CAF50;
                    '>
                        🎯 Final Score: {combine_score}
                    </div>
                """, unsafe_allow_html=True)
else:
    st.warning("Please upload a video file.")
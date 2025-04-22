import os
import sys
import streamlit as st
import tempfile
import numpy as np
import pandas as pd
import joblib
import langcodes
import matplotlib.pyplot as plt
import librosa
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

# Add necessary paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'tone_analysis_dashboard'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'speech_score'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'stress_analysis_refactored'))

# Import all functions from the respective modules
from preprocess_function import (
    extract_audio, preprocess_audio, predict_emotion,
    predict_fluency_level, get_emotion_interview_score,
    predict_personality, get_personality_interview_score,
    fluent_feature_extraction
)
from function_class import (
    convert_video_to_audio, transcribe_audio, preprocess_text,
    translate_to_indonesian, load_model_and_resources,
    predict_stress, predict_emotion as predict_text_emotion
)
from Function import get_video_duration, process_emotions, process_eye_gaze

# Models & score dictionaries
emotion_scores = {'Happy': 10, 'Neutral': 8, 'Surprise': 7, 'Sad': 5, 'Angry': 4, 'Disgust': 2, 'Fear': 1}
eye_gaze_scores = {'Forward Look': 10, 'Side Look (Left + Right)': 6, 'Right Look': 6, 'Left Look ': 6, 'Close Look': 3}

# Streamlit page
st.set_page_config(layout="wide")
st.title("AI Interview Evaluation Dashboard")

# Upload video or audio
uploaded_file = st.file_uploader("Upload a video or audio file", type=["mp4", "avi", "mov", "mkv", "mp3", "wav"])
if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    media_path = tfile.name

    # Extract audio
    if uploaded_file.type.startswith("video"):
        st.video(media_path)
        audio_path = extract_audio(media_path)
    else:
        audio_path = media_path

    # Tabs for each section
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Emotion + Fluency",
        "Personality Analysis",
        "Text-Based Emotion & Stress",
        "Facial Expression + Eye Gaze",
        "Final Score"
    ])

    # ========== TAB 1 ==========
    with tab1:
        try:
            features = preprocess_audio(audio_path)
            fluent_input = np.expand_dims(fluent_feature_extraction(audio_path), axis=0)
            emotion_le = joblib.load('tone_analysis_dashboard/emotion_model/emotion_label_encoder.joblib')
            emotion_scaler = joblib.load('tone_analysis_dashboard/emotion_model/emotion_feature_scaler.joblib')
            emotion_results = predict_emotion(features, emotion_scaler, emotion_le)
            fluent_labels = ['Low', 'Intermediate', 'High']
            fluent_prediction = predict_fluency_level(fluent_input)[0]
            fluent_level = fluent_labels[fluent_prediction]

            st.header("🎭 Emotion Analysis")
            for model, scores in emotion_results.items():
                fig = px.pie(values=[s * 100 for s in scores], names=emotion_le.classes_,
                            title="Emotion Distribution", color=emotion_le.classes_,
                            color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig)

                for emotion, score in zip(emotion_le.classes_, scores):
                    st.write(f"{emotion}: {score*100:.2f}%")

                most_likely_emotion = emotion_le.classes_[np.argmax(scores)]
                confidence = np.max(scores) * 100
                consistency = np.std(scores) * 100
                st.write(f"Most Likely: {most_likely_emotion}")
                st.write(f"Confidence: {confidence:.2f}%")
                st.write(f"Consistency: {consistency:.2f}%")
                st.write(f"Fluency Level: {fluent_level}")
                st.success(f"Emotion Interview Score: {get_emotion_interview_score(most_likely_emotion, fluent_level)} / 10")

            # Feature visualizations
            st.divider()
            st.header("Extracted Audio Features")
            st.write(features)
            st.write("Shape:", features.shape)

            y, sr = librosa.load(audio_path, sr=None)
            fig, ax = plt.subplots()
            librosa.display.waveshow(y, sr=sr, ax=ax)
            st.pyplot(fig)

            fig, ax = plt.subplots()
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            ax.set(title='Mel-frequency spectrogram')
            st.pyplot(fig)
            interview_score_tab1 = get_emotion_interview_score(most_likely_emotion, fluent_level)
        except:
            interview_score_tab1 = 0


    # ========== TAB 2 ==========
    with tab2:
        try:
            st.header("🧠 Personality Trait Analysis")
            scaler = joblib.load('tone_analysis_dashboard/personality_model2/personality_feature_scaler.joblib')
            personality_scores = predict_personality(features, scaler)
            traits = ['Neuroticism', 'Extraversion', 'Agreeableness', 'Conscientiousness', 'Openness']
            fig = px.line_polar(r=personality_scores, theta=traits, line_close=True)
            st.plotly_chart(fig)

            fig = go.Figure()
            fig.add_trace(go.Bar(y=traits, x=[1]*len(traits), orientation='h',
                                marker=dict(color='lightgray'), hoverinfo='none'))
            fig.add_trace(go.Bar(y=traits, x=personality_scores, orientation='h',
                                marker=dict(color='orange'), text=[f"{v*100:.1f}%" for v in personality_scores],
                                textposition='inside'))
            fig.update_layout(barmode='overlay', title="Traits (%)", xaxis_range=[0, 1])
            st.plotly_chart(fig)
            st.success(f"Personality Interview Score: {get_personality_interview_score(personality_scores) * 2} / 10")


            interview_score_tab2 = get_personality_interview_score(personality_scores) * 2
        except:
            interview_score_tab2 = 0


    # ========== TAB 3 ==========
    with tab3:
        try:
            st.header("🗣️ Text-Based Emotion & Stress Analysis")
            audio_file = convert_video_to_audio(media_path)
            transcript = transcribe_audio(audio_file)
            text = transcript["text"]
            lang = transcript["language"]

            st.write("Detected Language:", langcodes.get(lang).display_name())
            st.write("Transcript:", text)

            if lang != "id":
                text = translate_to_indonesian(text)

            preprocessed = preprocess_text(text)
            model, tokenizer, encoder = load_model_and_resources()
            stress = predict_stress(preprocessed, model, tokenizer, encoder)
            emotion = predict_text_emotion(preprocessed)

            NEG = ['anger', 'fear', 'sadness']
            POS = ['happy', 'love']
            if stress == 1 and emotion in NEG:
                text_score = 1
            elif stress == 1:
                text_score = 3
            elif stress == 0 and emotion in NEG:
                text_score = 2
            elif stress == 0:
                text_score = 4
            else:
                text_score = 0

            final_score = text_score * 2  # scale to 10

            st.write(f"Stress Detected: {'Yes' if stress else 'No'}")
            st.write(f"Emotion Detected: {emotion}")
            st.success(f"Combined Stress + Emotion Score: {final_score:.2f} / 10")

            interview_score_tab3 = final_score
        except:
            interview_score_tab3 = 0



    # ========== TAB 4 ==========
    with tab4:
        try:
            st.header("😶 Facial Expression & Eye Gaze Analysis")
            expression_model = load_model('stress_analysis_refactored/Model/Emotions_Model.h5')
            gaze_model = load_model('stress_analysis_refactored/Model/EyeGaze_Model.h5')
            video_duration = get_video_duration(media_path)

            # ------------------ Facial Expression ------------------
            st.subheader("🎭 Facial Expression Analysis")
            try:
                emotion_counts, total_frames, max_emotion, average_emotion_score = process_emotions(
                    media_path, expression_model, video_duration
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
                    raise ValueError("No emotion data extracted.")

            except Exception as e:
                average_emotion_score = 0
                st.error(f"⚠️ Facial expression analysis failed: {e}")
                st.info(f"🎯 Facial Emotion Score: {average_emotion_score:.2f} / 10")

            # ------------------ Eye Gaze ------------------
            st.subheader("👀 Eye Gaze Analysis")
            try:
                eye_gaze_counts, total_eye_frames, max_eye_gaze, final_eye_gaze_score = process_eye_gaze(
                    media_path, gaze_model, video_duration
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
                    raise ValueError("No eye gaze data extracted.")

            except Exception as e:
                final_eye_gaze_score = 0
                st.error(f"⚠️ Eye gaze analysis failed: {e}")
                st.info(f"🎯 Eye Gaze Score: {final_eye_gaze_score:.2f} / 10")

            # ------------------ Final Combined ------------------
            st.subheader("📊 Final Combined Score")
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

            interview_score_tab4 = combined_score  # already computed in tab 4
        except:
            interview_score_tab4 = 0
        

    with tab5:
        # Compute overall average
        all_scores = [interview_score_tab1, interview_score_tab2, interview_score_tab3, interview_score_tab4]
        final_average_score = round(np.mean(all_scores), 2)

        # Display
        st.header("🏆 Overall Interview Final Score")
        st.markdown(f"""
        <div style='
            background-color: #e6f2ff;
            padding: 18px;
            border-radius: 10px;
            text-align: center;
            font-size: 30px;
            color: #1a1a1a;
            font-weight: bold;
            border: 3px solid #009688;
        '>
            🌟 Final AI Interview Score: <span style='color:#007ACC'>{final_average_score} / 10</span>
        </div>
        """, unsafe_allow_html=True)

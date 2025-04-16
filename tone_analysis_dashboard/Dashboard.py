import streamlit as st
import cv2
import tempfile
import numpy as np
import pandas as pd
import librosa
import parselmouth
from typing import Tuple
from moviepy.editor import VideoFileClip
import joblib
import os
import tensorflow as tf
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from preprocess_function import extract_audio, preprocess_audio, predict_personality, predict_emotion, fluent_feature_extraction, predict_fluency_level, get_personality_interview_score, get_emotion_interview_score


    
# Streamlit app
st.title("AI Interview - Tone Analysis Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload a video or audio file", type=["mp4", "avi", "mov", "mkv", "mp3", "wav"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    # Check if the uploaded file is a video or audio
    if uploaded_file.type.startswith('video'):
        # Display video
        st.video(tfile.name)
        # Extract audio from video
        audiofile = extract_audio(tfile.name)
    else:
        # Save audio file directly
        audiofile = tfile.name

    # Show a processing message
    with st.spinner('Processing...'):
        # Preprocess audio
        features = preprocess_audio(audiofile)

        # feature for fluenccy level
        extracted_features = fluent_feature_extraction(audiofile)
        fluent_features = np.expand_dims(extracted_features, axis=0)

        # Load the label encoder and feature scaler
        traits = ['Neuroticism', 'Extraversion', 'Agreeableness', 'Conscientiousness', 'Openness']
        personality_scaler = joblib.load('personality_model2/personality_feature_scaler.joblib')

        # Load the label encoder and feature scaler
        emotion_le = joblib.load('emotion_model/emotion_label_encoder.joblib')
        emotion_scaler = joblib.load('emotion_model/emotion_feature_scaler.joblib')
        
        # Define the luency level label classes
        Fluent_label_classes = np.array(['Low','Intermediate','High'])

        # Predict personality traits
        personality_results = predict_personality(features, personality_scaler)
        
        # Predict emotions
        emotion_results = predict_emotion(features, emotion_scaler, emotion_le)

        #predict fluency level
        fluent_results = predict_fluency_level(fluent_features)


    # Create tabs for Personality Traits and Emotions Analysis
    tab1, tab2, tab3 = st.tabs(["Personality Traits Analysis", "Emotions Analysis", "Extracted Audio Features"])

    with tab1:
        # Display Personality Traits results
        st.header("Personality Traits Analysis")
        for trait, value in zip(traits, personality_results):
            print(f"{trait}: {value:.4f}")

        # Display the results in a radar chart
        fig = px.line_polar(r=personality_results, theta=traits, line_close=True)
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False
        )
        st.plotly_chart(fig)


        # Display the results in a horizontal bar chart with 100% bars
        st.header("Personality Traits Percentage")

        fig = go.Figure()

        # Background bars (white)
        fig.add_trace(go.Bar(
            y=traits,
            x=[1]*len(traits),  # 100% width
            orientation='h',
            marker=dict(color='rgba(255, 255, 255, 1)', line=dict(color='black', width=1)),
            showlegend=False,
            hoverinfo='none'
        ))

        # Foreground bars (actual percentage filled)
        fig.add_trace(go.Bar(
            y=traits,
            x=personality_results,  # Actual values
            orientation='h',
            marker=dict(color='orange', line=dict(color='black', width=1)),
            text=[f"{value*100:.2f}%" for value in personality_results],
            textposition='inside',
            showlegend=False
        ))

        # Update layout
        fig.update_layout(
            title="Personality Traits Percentage",
            xaxis_title="Percentage",
            yaxis_title="Traits",
            xaxis=dict(range=[0, 1]),
            barmode='overlay'  # Overlay the two bars
        )

        st.plotly_chart(fig)

        # Display the interview score
        st.header("Personality Interview Score")
        st.write(f"Personality Interview Score: {get_personality_interview_score(personality_results)}")





    with tab2:
        # Display Emotions results
        st.header("Emotions Analysis")
        for model, scores in emotion_results.items():
            st.subheader(model)
            if len(emotion_le.classes_) == len(scores):
                for emotion, score in zip(emotion_le.classes_, scores):
                    st.write(f"{emotion}: {score * 100:.2f}%")

                # Plot pie chart with consistent colors
                fig = px.pie(values=[s * 100 for s in scores], names=emotion_le.classes_, title=f"{model} Emotions",
                             color=emotion_le.classes_, color_discrete_sequence=px.colors.qualitative.Plotly)
                st.plotly_chart(fig)

                most_likely_emotion = emotion_le.classes_[np.argmax(scores)]
                st.write(f"Results : The model predicts that this candidate is more likely to be {most_likely_emotion}.")

                # Display confidence and consistency metrics
                confidence = np.max(scores) * 100
                consistency = np.std(scores) * 100
                st.write(f"Confidence: {confidence:.2f}%")
                st.write(f"Consistency: {consistency:.2f}%")
            else:
                st.error("Mismatch between emotions and scores. Check model output!")

            # Display Fluency Level results
            st.header("Fluency Level Analysis")
            st.write(f"Predicted Fluency Level: {Fluent_label_classes[fluent_results[0]]}")

            # Display Emotion Interview Score
            st.header("Emotion Interview Score")
            st.write(f"Emotion Interview Score: {get_emotion_interview_score(most_likely_emotion,Fluent_label_classes[fluent_results[0]])}")



    with tab3:
        # Display the extracted audio features
        st.header("Extracted Audio Features")
        st.write(features)
        st.write("Shape of the features:", features.shape)

        # Visualize the audio waveform
        st.header("Audio Waveform")
        y, sr = librosa.load(audiofile, sr=None)
        fig, ax = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set(title='Waveform of the Audio')
        st.pyplot(fig)

        # Visualize the spectrogram
        st.header("Spectrogram")
        fig, ax = plt.subplots()
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title='Mel-frequency spectrogram')
        st.pyplot(fig)
import streamlit as st
import tempfile
import requests
import numpy as np
import pandas as pd
import joblib
import langcodes
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import librosa

st.set_page_config(layout="wide")
st.title("AI Interview Evaluation Dashboard (from JSON)")

# 1️⃣ Let the user enter your API URL and upload a file
api_url = st.text_input("Analysis API endpoint", "http://localhost:5001/analyze")
uploaded_file = st.file_uploader("Upload your video/audio file", type=["mp4","avi","mov","mkv","mp3","wav"])

if api_url and uploaded_file:
    # POST to your Flask API
    with st.spinner("Sending to analysis API…"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        resp = requests.post(api_url, files=files)
    if not resp.ok:
        st.error(f"API error: {resp.status_code} {resp.text}")
        st.stop()

    data = resp.json()

    # load the same label‑encoder so we know the emotion names
    le = joblib.load("tone_analysis_dashboard/emotion_model/emotion_label_encoder.joblib")
    emotion_classes = list(le.classes_)

    # set up the five tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Emotion + Fluency",
        "Personality Analysis",
        "Text Emotion & Stress",
        "Facial Expression + Eye Gaze",
        "Final Score"
    ])

    # ─── TAB 1: Emotion + Fluency ──────────────────────────────────────────────
    with tab1:
        st.header("🎭 Emotion Analysis + Fluency Level")

        # Pie chart of the ensemble probabilities
        for model_name, scores in data["emotion_results"].items():
            fig = px.pie(
                values=scores,
                names=emotion_classes,
                title=f"{model_name} Distribution",
                color=emotion_classes,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig)

        # show each %  
        st.write("**Probabilities:**")
        for emo, p in zip(emotion_classes, data["emotion_results"][model_name]):
            st.write(f"- {emo}: {p*100:.2f}%")

        # Most likely
        st.write(f"**Most Likely Emotion:** {data['most_likely_emotion'].capitalize()}")
        # Compute confidence & consistency client‑side:
        probs = np.array(data["emotion_results"][model_name])
        st.write(f"**Confidence:** {probs.max()*100:.2f}%")
        st.write(f"**Consistency:** {probs.std()*100:.2f}%")

        # fluency
        st.write(f"**Fluency Level:** {data['fluent_level']}")

        # final interview score (tab1)
        st.success(f"🎯 Emotion Interview Score: {data['interview_score_tab1']:.2f} / 10")

    # ─── TAB 2: Personality ────────────────────────────────────────────────────
    with tab2:
        st.header("🧠 Personality Trait Analysis")

        scores = data["personality_scores"]
        traits = ["Neuroticism","Extraversion","Agreeableness","Conscientiousness","Openness"]

        # radar
        fig = px.line_polar(r=scores, theta=traits, line_close=True, 
                            title="Personality Radar")
        st.plotly_chart(fig)

        # bar % overlay
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(y=traits, x=[1]*5, orientation="h",
                              marker=dict(color="lightgray"), showlegend=False))
        fig2.add_trace(go.Bar(y=traits, x=scores, orientation="h",
                              marker=dict(color="orange"),
                              text=[f"{v*100:.1f}%" for v in scores],
                              textposition="inside", showlegend=False))
        fig2.update_layout(barmode="overlay", xaxis_range=[0,1],
                           title="Traits %")
        st.plotly_chart(fig2)

        st.success(f"🎯 Personality Interview Score: {data['interview_score_tab2']:.2f} / 10")

    # ─── TAB 3: Text‑Based Emotion & Stress ───────────────────────────────────
    with tab3:
        st.header("🗣️ Text-Based Emotion & Stress Analysis")
        st.write(f"**Detected Language:** {langcodes.get(data['detected_language']).display_name()}")
        st.write(f"**Transcript:** {data['transcript']}")

        st.write(f"**Stress Detected:** {'Yes' if data['stress_detected'] else 'No'}")
        st.write(f"**Text Emotion:** {data['text_emotion'].capitalize()}")

        st.success(f"🎯 Combined Text Score: {data['interview_score_tab3']:.2f} / 10")

    # ─── TAB 4: Facial Expression + Eye Gaze ─────────────────────────────────
    with tab4:
        st.header("😶 Facial Expression & Eye Gaze Analysis")

        # Emotion distribution table & chart
        df_em = pd.DataFrame(data["emotion_distribution"])
        st.subheader("🎭 Facial Emotion Distribution")
        fig3, ax3 = plt.subplots()
        ax3.bar(df_em["Emotions"], df_em["Frames"],
                color=["#FF9999","#66B2FF","#99FF99","#FFCC99","#FFD700","#87CEFA","#90EE90"])
        ax3.set_ylabel("Frames"); ax3.set_xlabel("Emotions")
        ax3.set_title("Emotion Distribution")
        plt.xticks(rotation=45)
        st.pyplot(fig3)

        st.write("### Distribution Table")
        st.table(df_em)

        st.write(f"**Predominant Emotion:** {data['max_emotion']}")
        st.write(f"**Facial Emotion Score:** {data['average_emotion_score']:.2f} / 10")

        # Eye gaze
        df_gz = pd.DataFrame(data["eye_gaze_distribution"])
        st.subheader("👀 Eye Gaze Distribution")
        fig4, ax4 = plt.subplots()
        ax4.bar(df_gz["Eye Gaze"], df_gz["Frames"],
                color=["#FF9999","#66B2FF","#99FF99"])
        ax4.set_ylabel("Frames"); ax4.set_xlabel("Eye Gaze")
        ax4.set_title("Eye Gaze Distribution")
        plt.xticks(rotation=45)
        st.pyplot(fig4)

        st.write("### Distribution Table")
        st.table(df_gz)

        st.write(f"**Predominant Eye Gaze:** {data['max_eye_gaze']}")
        st.write(f"**Eye Gaze Score:** {data['final_eye_gaze_score']:.2f} / 10")

        # combined tab4 score
        st.success(f"🎯 Facial+Gaze Score: {data['interview_score_tab4']:.2f} / 10")

    # ─── TAB 5: Final Score ───────────────────────────────────────────────────
    with tab5:
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
            🌟 Final AI Interview Score: <span style='color:#007ACC'>{data['final_average_score']:.2f} / 10</span>
        </div>
        """, unsafe_allow_html=True)

# app.py
import os
import sys
import tempfile
import numpy as np
import pandas as pd
import joblib
import langcodes
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

# add your module paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'tone_analysis_dashboard'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'speech_score'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'stress_analysis_refactored'))

# imports from your custom modules
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

# pre-load heavy models
emotion_model = load_model('stress_analysis_refactored/Model/Emotions_Model.h5')
gaze_model    = load_model('stress_analysis_refactored/Model/EyeGaze_Model.h5')

app = Flask(__name__)


def df_to_records(df: pd.DataFrame):
    """Convert a DataFrame into a list of JSON-safe dicts."""
    records = df.to_dict(orient="records")
    for rec in records:
        for k, v in rec.items():
            # cast any numpy scalar to Python
            if isinstance(v, (np.generic, np.ndarray)):
                rec[k] = v.item() if np.isscalar(v) else v.tolist()
    return records

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "no file part"}), 400
    f = request.files['file']
    suffix = os.path.splitext(f.filename)[1] or '.mp4'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(f.read())
        media_path = tmp.name

    # 1. Emotion + Fluency
    audio_path = extract_audio(media_path) if f.mimetype.startswith('video') else media_path
    features = preprocess_audio(audio_path)
    fluent_input = np.expand_dims(fluent_feature_extraction(audio_path), axis=0)
    emotion_le = joblib.load('tone_analysis_dashboard/emotion_model/emotion_label_encoder.joblib')
    emotion_scaler = joblib.load('tone_analysis_dashboard/emotion_model/emotion_feature_scaler.joblib')
    emotion_results = predict_emotion(features, emotion_scaler, emotion_le)
    fluent_prediction = predict_fluency_level(fluent_input)[0]
    fluent_labels = ['Low','Intermediate','High']
    fluent_level = fluent_labels[fluent_prediction]
    model_name, scores = next(iter(emotion_results.items()))
    most_likely_emotion = emotion_le.classes_[np.argmax(scores)]
    interview_score_tab1 = float( get_emotion_interview_score(most_likely_emotion, fluent_level))
    

    # 2. Personality Analysis
    scaler2 = joblib.load('tone_analysis_dashboard/personality_model2/personality_feature_scaler.joblib')
    personality_scores = predict_personality(features, scaler2)
    interview_score_tab2 = float(get_personality_interview_score(personality_scores) * 2)

    # 3. Text‑Based Emotion & Stress
    audio_for_text = convert_video_to_audio(media_path)
    transcript = transcribe_audio(audio_for_text)
    text = transcript["text"]
    lang = transcript["language"]
    if lang != "id":
        text = translate_to_indonesian(text)
    preprocessed_text = preprocess_text(text)
    model_t, tok, enc = load_model_and_resources()
    stress = predict_stress(preprocessed_text, model_t, tok, enc)
    text_emotion = predict_text_emotion(preprocessed_text)
    # map to 0–4 then scale ×2 to get 0–10
    if stress == 1 and text_emotion in ['anger','fear','sadness']:
        ts = 1
    elif stress == 1:
        ts = 3
    elif stress == 0 and text_emotion in ['anger','fear','sadness']:
        ts = 2
    elif stress == 0:
        ts = 4
    else:
        ts = 0
    interview_score_tab3 = float(ts * 2)

    # 4. Facial Expression + Eye Gaze
    video_duration = get_video_duration(media_path)
    ec, tfc, max_em, avg_em_score = process_emotions(media_path, emotion_model, video_duration)
    gc, tgc, max_gz, gaze_score = process_eye_gaze(media_path, gaze_model, video_duration)
    interview_score_tab4 = float((avg_em_score + gaze_score) / 2)

    # 5. Final average
    all_scores = [interview_score_tab1, interview_score_tab2, interview_score_tab3, interview_score_tab4]
    final_average_score = float(np.mean([
        interview_score_tab1,
        interview_score_tab2,
        interview_score_tab3,
        interview_score_tab4
    ]))

    print(f"Final average score: {final_average_score:.2f}")

    return jsonify({
        # Tab 1
        "emotion_results": { m: [float(s) for s in scores] for m, scores in emotion_results.items() },
        "most_likely_emotion": most_likely_emotion,
        "fluent_level": fluent_level,
        "interview_score_tab1": interview_score_tab1,

        # Tab 2
        "personality_scores": [float(v) for v in personality_scores],
        "interview_score_tab2": interview_score_tab2,

        # Tab 3
        "transcript": text,
        "detected_language": lang,
        "stress_detected": bool(stress),
        "text_emotion": text_emotion,
        "interview_score_tab3": interview_score_tab3,

        # Tab 4
        "emotion_distribution": df_to_records(ec), # ec returned by process_emotions()
        "max_emotion": max_em,                     # max_em returned by process_emotions()
        "average_emotion_score": float(avg_em_score),  # avg_em_score from process_emotions()
        "eye_gaze_distribution": df_to_records(gc),    # gc returned by process_eye_gaze()
        "max_eye_gaze": max_gz,                        # max_gz from process_eye_gaze()
        "final_eye_gaze_score": float(gaze_score),     # gaze_score from process_eye_gaze()
        "interview_score_tab4": interview_score_tab4,  # the averaged score

        # Final
        "final_average_score": final_average_score
    })

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5001)

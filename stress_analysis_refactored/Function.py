import cv2
import numpy as np
import pandas as pd

# Define scores for each emotion
EMOTION_SCORES = {
    'Happy': 10,
    'Neutral': 8.33,
    'Surprise': 6.67,
    'Sad': 5,
    'Angry': 3.33,
    'Disgust': 1.67,
    'Fear': 0
}

# Define scores for each eye gaze type
EYE_GAZE_SCORES = {
    'Forward Look': 10,
    'Side Look (Left + Right)': 5,
    'Right Look': 5,
    'Left Look ': 5,
    'Close Look': 0
}

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    success = True
    count = 0
    while success:
        success, frame = cap.read()
        if not success or frame is None:
            print("Video fully processed. All frames extracted.")
            break
        frames.append(frame)
        count += 1
    cap.release()
    print(f"Total frames extracted: {count}")
    return frames

def detect_and_crop_face(frames):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    cropped_faces = []
    for frame in frames:
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            cropped_face = frame[y:y+h, x:x+w]
            cropped_faces.append(cropped_face)
    return cropped_faces

def detect_and_crop_eye(frames):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    cropped_eyes = []
    for frame in frames:
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            roi_gray = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
            if len(eyes) > 0:
                # Sort eyes by x-coordinate to get the right eye
                eyes = sorted(eyes, key=lambda e: e[0], reverse=True)
                right_eye = eyes[0]
                (ex, ey, ew, eh) = right_eye
                cropped_eye = roi_gray[ey:ey+eh, ex:ex+ew]
                cropped_eyes.append(cropped_eye)
    return cropped_eyes

def resize_images(images, target_size):
    resized_images = [cv2.resize(image, target_size, interpolation=cv2.INTER_AREA) for image in images]
    return resized_images

def convert_to_grayscale(images):
    grayscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
    return grayscale_images

def normalize_images(images):
    normalized_images = [image / 255.0 for image in images]
    return normalized_images

def preprocess_fer(video_path):
    frames = load_video(video_path)
    frames = convert_to_grayscale(frames)
    faces = detect_and_crop_face(frames)
    resized_faces = resize_images(faces, (48, 48))
    normalized_faces = normalize_images(resized_faces)
    return normalized_faces

def preprocess_eye_gaze(video_path):
    frames = load_video(video_path)
    frames = convert_to_grayscale(frames)
    eyes = detect_and_crop_eye(frames)
    resized_eyes = resize_images(eyes, (60, 60))
    normalized_eyes = normalize_images(resized_eyes)
    return normalized_eyes

def get_video_duration(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)  # Frames per second
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)  # Total number of frames
    duration = frame_count / fps  # Duration in seconds
    video.release()
    return duration

def empty_emotion_df():
    return pd.DataFrame(columns=['Emotions','Frames','Percentage (%)','Duration (seconds)'])

def empty_gaze_df():
    return pd.DataFrame(columns=['Eye Gaze','Frames','Percentage (%)','Duration (seconds)'])


def process_emotions(video_path, emotion_model, video_duration):
    # 1) extract, crop, normalize
    faces = preprocess_fer(video_path)
    if len(faces) == 0:
        return empty_emotion_df(), 0, None, 0.0

    X = np.array(faces)
    try:
        # silent predict
        preds = emotion_model.predict(X, verbose=0)
        idxs = np.argmax(preds, axis=1)
    except Exception:
        return empty_emotion_df(), 0, None, 0.0

    labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
    counts = pd.Series(idxs).value_counts().sort_index()
    counts.index = [labels[i] for i in counts.index]
    df = counts.reset_index()
    df.columns = ['Emotions','Frames']

    total = int(df['Frames'].sum())
    df['Percentage (%)']     = df['Frames'] / total * 100
    df['Duration (seconds)'] = df['Percentage (%)'] / 100 * video_duration
    df = df.sort_values('Duration (seconds)', ascending=False).reset_index(drop=True)

    # pick predominant (skip neutral if it dominates)
    max_em = df.loc[df['Frames'].idxmax(), 'Emotions']
    if max_em == 'Neutral':
        alt = df[df['Duration (seconds)']>0.5]
        alt = alt[alt['Emotions']!='Neutral']
        if not alt.empty:
            max_em = alt.loc[alt['Duration (seconds)'].idxmax(), 'Emotions']

    # average score over emotions >0.5s
    filt = df[df['Duration (seconds)']>0.5]
    if not filt.empty:
        filt = filt.copy()
        filt['Score'] = filt['Emotions'].map(EMOTION_SCORES)
        avg_score = float(filt['Score'].mean())
    else:
        avg_score = 0.0

    return df, total, max_em, avg_score


def process_eye_gaze(video_path, eye_gaze_model, video_duration):
    eyes = preprocess_eye_gaze(video_path)
    if len(eyes) == 0:
        return empty_gaze_df(), 0, None, 0.0

    X = np.array(eyes)
    try:
        preds = eye_gaze_model.predict(X, verbose=0)
        idxs = np.argmax(preds, axis=1)
    except Exception:
        return empty_gaze_df(), 0, None, 0.0

    labels = ['Close Look','Forward Look','Left Look','Right Look']
    counts = pd.Series(idxs).value_counts().sort_index()
    counts.index = [labels[i] for i in counts.index]
    df = counts.reset_index()
    df.columns = ['Eye Gaze','Frames']

    # combine Left+Right
    if {'Left Look','Right Look'}.issubset(df['Eye Gaze'].values):
        left  = df.loc[df['Eye Gaze']=='Left Look','Frames'].iat[0]
        right = df.loc[df['Eye Gaze']=='Right Look','Frames'].iat[0]
        df = df[~df['Eye Gaze'].isin(['Left Look','Right Look'])]
        df = pd.concat([df, pd.DataFrame({
            'Eye Gaze': ['Side Look (Left + Right)'],
            'Frames': [left+right]
        })], ignore_index=True)

    total = int(df['Frames'].sum())
    df['Percentage (%)']     = df['Frames'] / total * 100
    df['Duration (seconds)'] = df['Percentage (%)'] / 100 * video_duration
    df = df.sort_values('Duration (seconds)', ascending=False).reset_index(drop=True)

    max_gz = df.loc[df['Frames'].idxmax(), 'Eye Gaze']
    score = float(EYE_GAZE_SCORES.get(max_gz, 0))

    return df, total, max_gz, score
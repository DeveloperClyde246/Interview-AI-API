import streamlit as st 
import re
import os
import whisper
import json 
import torch
import moviepy as mp
from moviepy.editor import VideoFileClip 
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from transformers import pipeline
from langdetect import detect
from nltk.tokenize import word_tokenize
import pandas as pd
import string
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import torch

import nltk
nltk.download('punkt_tab')

# Function to convert video to audio
def convert_video_to_audio(video_file, output_audio_name="output_audio.wav"):
    try:
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(video_file.read())

        video = VideoFileClip(temp_video_path)  # ✅ updated call
        audio = video.audio
        audio.write_audiofile(output_audio_name)
        video.close()
        os.remove(temp_video_path)

        return output_audio_name

    except Exception as e:
        st.error(f"Error occurred during audio extraction: {str(e)}")
        return None

# Whisper transcription
def transcribe_audio(audio_path):
    model = whisper.load_model("large-v3")  # Change model size if needed
    result = model.transcribe(audio_path)
    return result

# Preprocess text
def preprocess_text(text):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Load Indonesian stopwords from JSON file

    stopwords_path = os.path.join(script_dir, "stopwords-id.json")
    with open(stopwords_path, "r", encoding="utf-8") as file:
        indonesian_stopwords = set(json.load(file))  # Convert to set for faster lookup

    # Load slang words dictionary from JSON file
    combined_slang_path = os.path.join(script_dir, "combined_slang_words.txt")
    with open(combined_slang_path, "r", encoding="utf-8") as file:
        slang_dict = json.load(file)  # Load JSON directly into a dictionary

    # Initialize Sastrawi stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    if pd.isna(text) or text.strip() == "":  # Handle missing and empty values
        return ""

    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenization

    # Replace slang words
    tokens = [slang_dict.get(word, word) for word in tokens]

    # Remove stopwords
    tokens = [word for word in tokens if word not in indonesian_stopwords]

    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]

    return " ".join(tokens)

# # Remove Indonesian stopwords
# def remove_stopwords(text, stopwords_path="C:/Users/Admin/OneDrive/Desktop/speech-score/stopwords-id.json"):
#     with open(stopwords_path, "r") as f:
#         stopwords = json.load(f)
#     words = text.split()
#     filtered_words = [word for word in words if word not in stopwords]
#     return ' '.join(filtered_words)

# # Convert Indonesian slang words
# def convert_slang(text, slang_path="C:/Users/Admin/OneDrive/Desktop/speech-score/combined_slang_words.txt"):
#     # Load the JSON file
#     with open(slang_path, "r", encoding="utf-8") as f:
#         slang_dict = json.load(f)  # Load as a dictionary

#     # Replace words in the text based on the slang dictionary
#     words = text.split()
#     converted_words = [slang_dict.get(word, word) for word in words]
#     return ' '.join(converted_words)

# def stem_text(text):
#     # Create the stemmer
#     stem_factory = StemmerFactory()
#     stemmer = stem_factory.create_stemmer()

#     # Tokenize the text (split into words)
#     tokens = text.split()
    
#     # Apply stemming to the tokens
#     stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
#     # Return the stemmed text (joined back into a string)
#     return ' '.join(stemmed_tokens)

# Translate text to Indonesian
def translate_to_indonesian(sentence):
    try:
        translation_pipeline = pipeline("translation_en_to_id", model="Helsinki-NLP/opus-mt-en-id")
        full_sentence_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-en-id")
        
        sentence = str(sentence)  # Ensure input is string
        detected_lang = detect(sentence)
        
        if detected_lang == "en":
            return full_sentence_pipeline(sentence)[0]['translation_text']
        
        elif detected_lang == "id":
            translated_words = []
            for word in sentence.split():
                try:
                    if detect(word) == "en":
                        translation = translation_pipeline(word)[0]['translation_text']
                        translated_words.append(translation if translation else word)
                    else:
                        translated_words.append(word)
                except:
                    translated_words.append(word)
            return " ".join(translated_words)
        
        return sentence
    
    except:
        return str(sentence)  # Fallback to original as string

def load_model_and_resources():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Load BiLSTM model
    bilstm_path = os.path.join(script_dir, "bilstm_model.h5")
    model = tf.keras.models.load_model(bilstm_path)
    
    # 2. Load tokenizer - FIXED VERSION
    tokenizer_path = os.path.join(script_dir, "tokenizer.json")
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer = tokenizer_from_json(f.read())  # Pass the raw JSON string
    
    # 3. Load label encoder
    label_encoder_path = os.path.join(script_dir, "label_encoder.npy")
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(label_encoder_path, allow_pickle=True)
    
    return model, tokenizer, label_encoder

def predict_stress(speech_text, model, tokenizer, label_encoder, maxlen=50):
    # Tokenize and pad the input text
    sequence = tokenizer.texts_to_sequences([speech_text])
    padded_sequence = pad_sequences(sequence, maxlen=maxlen, padding='post')  # Adjust maxlen if needed

    # Predict
    predictions = model.predict(padded_sequence)
    predicted_labels = label_encoder.inverse_transform((predictions > 0.5).astype(int).flatten())
    return predicted_labels[0]

# First modify the predict_emotion function to return config
# Function to predict emotions
def predict_emotion(text):
    # Load the pretrained model and tokenizer
    pretrained_name = "StevenLimcorn/indonesian-roberta-base-emotion-classifier"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_name)
    model.eval()  # Set model to evaluation mode
    if not isinstance(text, str) or text.strip() == "":
        return "Unknown"  # Handle empty or non-string inputs

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_label_id = torch.argmax(probabilities, dim=-1).item()
    
    # Convert numerical ID to label name
    config = AutoConfig.from_pretrained(pretrained_name)
    id2label = config.id2label  # Dictionary that maps numbers to labels
    predicted_label = id2label.get(predicted_label_id, "Unknown")
    return predicted_label


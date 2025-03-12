import cv2
from deepface import DeepFace
from transformers import pipeline
import speech_recognition as sr
import numpy as np
from src.utils import softmax, load_config
import time
import streamlit as st

config = load_config()

class EmotionDetector:
    def __init__(self):
        self.video_detector = DeepFace
        self.text_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = config["emotion_detection"]["speech_threshold"]
        self.emotions = ["happy", "sad", "angry", "fear", "disgust", "neutral"]

    def detect_video_emotion(self, frame):
        try:
            result = self.video_detector.analyze(frame, actions=['emotion'], enforce_detection=False)
            return result[0]['dominant_emotion']
        except Exception as e:
            print(f"Video detection error: {e}")
            return "neutral"

    def detect_text_emotion(self, text):
        if not text:
            return "neutral"
        try:
            results = self.text_classifier(text)[0]
            scores = {res['label']: res['score'] for res in results}
            return max(scores, key=scores.get).lower()
        except Exception as e:
            print(f"Text detection error: {e}")
            return "neutral"

    def detect_speech_emotion(self, device_index=None):
        print("Recording audio...")
        try:
            with sr.Microphone(device_index=device_index) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("Microphone ready, speak now...")
                audio = self.recognizer.listen(source, timeout=config["emotion_detection"]["speech_timeout"], phrase_time_limit=5)
                print("Processing audio...")
                text = self.recognizer.recognize_google(audio)
                return self.detect_text_emotion(text)
        except sr.WaitTimeoutError:
            print("No speech detected.")
            return "neutral"
        except sr.UnknownValueError:
            print("Audio not understood.")
            return "neutral"
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return "neutral"

    def analyze_video_duration(self, duration, callback):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return "neutral", []
        
        emotions = []
        start_time = time.time()
        
        while cap.isOpened() and (time.time() - start_time) < duration:
            ret, frame = cap.read()
            if not ret:
                break
            emotion = self.detect_video_emotion(frame)
            emotions.append(emotion)
            callback(frame, emotion)  # Update UI in real-time
            if not st.session_state.get("analyzing", False):
                break
        
        cap.release()
        if emotions:
            # Return the most frequent emotion
            return max(set(emotions), key=emotions.count), emotions
        return "neutral", emotions

    def compute_stress_score(self, video_emotion, text_emotion, speech_emotion):
        stress_emotions = ["sad", "angry", "fear", "disgust"]
        weights = [0.4, 0.3, 0.3]
        scores = [
            1.0 if video_emotion in stress_emotions else 0.0,
            1.0 if text_emotion in stress_emotions else 0.0,
            1.0 if speech_emotion in stress_emotions else 0.0
        ]
        return sum(w * s for w, s in zip(weights, scores))
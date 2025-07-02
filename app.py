import cv2
from deepface import DeepFace
from gtts import gTTS
import pygame
import os
import time

# Initialize webcam
cap = cv2.VideoCapture(0)
last_spoken = 0  # Time since last speech
speak_interval = 5  # Seconds between each voice output

# Initialize pygame for audio playback
pygame.mixer.init()

def speak(text):
    tts = gTTS(text=text)
    tts.save("speech.mp3")
    pygame.mixer.music.load("speech.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue
    os.remove("speech.mp3")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.analyze(frame, actions=['emotion', 'age', 'gender'], enforce_detection=False)[0]
        overlay = f"{result['dominant_emotion']}, {result['age']}, {result['gender']}"
        cv2.putText(frame, overlay, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Speak only every 5 seconds
        current_time = time.time()
        if current_time - last_spoken > speak_interval:
            speak_text = f"You look {result['dominant_emotion']} and approximately {result['age']} years old. Gender: {result['gender']}."
            speak(speak_text)
            last_spoken = current_time

    except Exception as e:
        print("Error:", e)

    cv2.imshow("DeepFace Cam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

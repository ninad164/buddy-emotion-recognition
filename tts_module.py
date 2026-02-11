# tts_module.py
import pyttsx3

# Initialize TTS engine once
engine = pyttsx3.init()
engine.setProperty("rate", 180)  # speaking speed

def speak(text: str):
    """Convert text to speech and play it."""
    if not text:
        return
    engine.say(text)
    engine.runAndWait()
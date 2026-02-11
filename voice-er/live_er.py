import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import torch
from transformers import pipeline
import tempfile

# Load Whisper-based emotion model
emotion_pipe = pipeline(
    task="audio-classification",
    model="firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3",
    device=0  # Use GPU
)

# Audio config
SAMPLING_RATE = 16000
CHANNELS = 1
DURATION = 3  # seconds

# Find ReSpeaker device
devices = sd.query_devices()
respeaker_index = None
for i, d in enumerate(devices):
    if "ReSpeaker" in d['name']:
        respeaker_index = i
        break

if respeaker_index is None:
    print("ReSpeaker Mic Array not found.")
    exit()

print(f"Using ReSpeaker Mic: {devices[respeaker_index]['name']}")

# Real-time loop
try:
    while True:
        print("🔴 Capturing 3 seconds...")
        audio = sd.rec(int(DURATION * SAMPLING_RATE),
                       samplerate=SAMPLING_RATE,
                       channels=CHANNELS,
                       dtype='int16',
                       device=respeaker_index)
        sd.wait()

        # Save to temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            wav.write(tmpfile.name, SAMPLING_RATE, audio)
            result = emotion_pipe(tmpfile.name)
            top = result[0]
            print(f"🧠 Emotion: {top['label']} ({top['score']:.2f})")
            print("-" * 40)

except KeyboardInterrupt:
    print("🛑 Stopped by user.")
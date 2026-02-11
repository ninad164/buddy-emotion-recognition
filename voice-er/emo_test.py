from transformers import pipeline

# Load the Whisper-based emotion model
emotion_pipe = pipeline(
    task="audio-classification",
    model="firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3",
    device=0  # Use GPU
)

# Run inference on a sample WAV file
result = emotion_pipe("test.wav")  # Replace with your actual file
print(result)
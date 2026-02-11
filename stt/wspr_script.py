from transformers import pipeline
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# Load Whisper Medium
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-medium",
    device=device,
    dtype=torch.float16,
    chunk_length_s=5,
    generate_kwargs={"language": "en", "task": "transcribe"}
)

# Test with a short audio file (16kHz mono WAV)
result = pipe("harvard.wav") 
print("Transcript:", result["text"])
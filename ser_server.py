from fastapi import FastAPI, UploadFile, File
from transformers import pipeline
import soundfile as sf
import librosa

app = FastAPI()

# Load Whisper Large-v3 fine-tuned for SER (firdhokk)
ser_pipe = pipeline(
    "audio-classification",
    model="firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3",
    device=0   # GPU (set -1 for CPU)
)

@app.post("/ser")
async def ser(file: UploadFile = File(...)):
    # Read uploaded audio file
    audio, sr = sf.read(file.file)
    # audio = np.array(audio).astype(np.float32)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Run SER
    results = ser_pipe(audio, sampling_rate=sr)
    top = max(results, key=lambda r: r["score"])

    return {
        "emotion": top["label"],
        "confidence": float(top["score"]),
        "raw": results
    }
# voice_module.py
from config import SAMPLE_RATE, SER_API_URL, DURATION, CHANNELS, LANGUAGE, BEAM_SIZE, COMPUTE_TYPE
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import requests
import soundfile as sf
import io

# Load STT model once
stt_model = WhisperModel("small.en", device="cuda", compute_type=COMPUTE_TYPE)

# --- Core Functions ---

def capture_audio():
    """Record audio from microphone after pressing Enter."""
    try:
        input("\n🎤 Press Enter to start recording...")
    except EOFError:
        return None

    print("🎙️ Recording...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=CHANNELS,
                   dtype='float32')
    sd.wait()
    print("✅ Recording complete")
    return np.squeeze(audio)

def transcribe_with_model(audio, model):
    """Helper to run Whisper transcription."""
    segments, _ = model.transcribe(audio,
                                   language=LANGUAGE,
                                   beam_size=max(1, BEAM_SIZE))
    return "".join(seg.text for seg in segments).strip() or "[Unintelligible or silent input]"

def run_stt(audio):
    """Speech-to-text with fallback if GPU OOM."""
    try:
        return transcribe_with_model(audio, stt_model)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("[STT] OOM. Retrying with Whisper Medium.")
            fallback = WhisperModel("medium.en", device="cuda", compute_type=COMPUTE_TYPE)
            return transcribe_with_model(audio, fallback)
        print(f"[STT] Error: {e}")
        return "[STT error]"

def run_ser(audio):
    """Send audio to SER API and return (emotion, confidence)."""
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    buf.seek(0)
    try:
        resp = requests.post(SER_API_URL,
                             files={"file": ("audio.wav", buf, "audio/wav")},
                             timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["emotion"], data["confidence"]
    except Exception as e:
        print(f"[SER API Error] {e}")
        return "neutral", 0.0
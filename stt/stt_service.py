import time
import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel

# ---- Config ----
SAMPLE_RATE = 16000          # Whisper expects 16k mono
CHUNK_DURATION = 3           # seconds per chunk (2–3s = low latency)
LANGUAGE = "en"              # force English to skip language detection
BEAM_SIZE = 1                # 1 for speed, increase for accuracy if you have headroom
COMPUTE_TYPE = "float16"     # try "int8_float16" if you want even more speed

# ---- Model ----
print("Loading Whisper Large-v2 model...")
model = WhisperModel("large-v2", device="cuda", compute_type=COMPUTE_TYPE)

# ---- VAD ----
vad = webrtcvad.Vad(2)  # 0-3, higher = more aggressive

def record_chunk(duration=CHUNK_DURATION, samplerate=SAMPLE_RATE):
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate,
                   channels=1, dtype="float32")
    sd.wait()
    return np.squeeze(audio)

def is_speech(audio_float32, sample_rate=SAMPLE_RATE):
    """Check if chunk contains speech using WebRTC VAD."""
    # Convert float32 numpy array to 16-bit PCM bytes
    audio_int16 = (audio_float32 * 32768).astype(np.int16)
    pcm_bytes = audio_int16.tobytes()
    # VAD works on 10, 20, or 30 ms frames
    frame_duration_ms = 30
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    for start in range(0, len(audio_int16) - frame_size, frame_size):
        frame = pcm_bytes[start*2:(start+frame_size)*2]
        if vad.is_speech(frame, sample_rate):
            return True
    return False

# Warm-up to load weights into GPU
_ = model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), language=LANGUAGE, beam_size=BEAM_SIZE)

print("Ready. Speak into the mic.")
while True:
    audio_chunk = record_chunk()
    ts = time.time()
    if not is_speech(audio_chunk):
        continue  # skip silent chunks

    segments, info = model.transcribe(audio_chunk, language=LANGUAGE, beam_size=BEAM_SIZE)
    text = "".join(seg.text for seg in segments).strip()
    if text:
        print(f"[{ts:.2f}] {text}")
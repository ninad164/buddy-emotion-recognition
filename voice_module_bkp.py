import sounddevice as sd
import numpy as np
import webrtcvad
import time
from faster_whisper import WhisperModel
from config import SER_API_URL, DURATION, SAMPLE_RATE, CHANNELS, LANGUAGE, BEAM_SIZE, COMPUTE_TYPE
import requests
import soundfile as sf

# Local SER pipeline (Whisper Large-v3 fine-tuned by firdhokk)
# from transformers import pipeline
# emotion_pipe = pipeline(
#     task="audio-classification",
#     model="firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3",
#     device=0   # GPU (set -1 for CPU)
# )

def get_ser_from_api(audio, sr):
    sf.write("temp.wav", audio, sr)
    with open("temp.wav", "rb") as f:
        response = requests.post(SER_API_URL, files={"file": f})
    data = response.json()
    return data["emotion"], data["confidence"], data["raw"]

try:
    stt_model = WhisperModel("medium.en", device="cuda", compute_type="int8_float16")
    print("[STT] Loaded Whisper Medium on GPU")
except RuntimeError as e:
    print(f"[STT] Medium failed to load on GPU ({e}). Falling back to Tiny.")
    stt_model = WhisperModel("tiny.en", device="cuda", compute_type=COMPUTE_TYPE)
    print("[STT] Loaded Whisper Tiny on GPU")

vad = webrtcvad.Vad(0)

def is_speech(audio_float32, sample_rate=SAMPLE_RATE):
    audio_int16 = (np.clip(audio_float32, -1.0, 1.0) * 32768).astype(np.int16)
    pcm_bytes = audio_int16.tobytes()
    frame_duration_ms = 30
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    for start in range(0, len(audio_int16) - frame_size, frame_size):
        frame = pcm_bytes[start*2:(start+frame_size)*2]
        if vad.is_speech(frame, sample_rate):
            return True
    return False

def capture_audio(device_index=None):
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='float32',
        device=device_index
    )
    sd.wait()
    audio = np.squeeze(audio)
    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak
    return audio

def run_stt(audio):
    try:
        segments, _ = stt_model.transcribe(
            audio,
            language=LANGUAGE,
            beam_size=max(1, BEAM_SIZE)
        )
        transcript = "".join(seg.text for seg in segments).strip()
        return transcript if transcript else "[Unintelligible or silent input]"
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("[STT] OOM during transcription. Retrying with Whisper Medium on GPU.")
            fallback_model = WhisperModel("medium.en", device="cuda", compute_type=COMPUTE_TYPE)
            segments, _ = fallback_model.transcribe(
                audio,
                language=LANGUAGE,
                beam_size=max(1, BEAM_SIZE)
            )
            transcript = "".join(seg.text for seg in segments).strip()
            return transcript if transcript else "[Unintelligible or silent input]"
        else:
            print(f"[STT] Error during transcription: {e}")
            return "[STT error]"

# def run_ser(audio):
    '''Run speech emotion recognition locally.'''
    results = emotion_pipe(audio, sampling_rate=SAMPLE_RATE)
    top = max(results, key=lambda r: r["score"])
    return top["label"], float(top["score"]), results

def process_voice(device_index=None):
    audio = capture_audio(device_index)
    if not is_speech(audio):
        return {
            "transcript": "",
            "speech_emotion": "neutral",
            "speech_conf": 0.0,
            "raw_emotions": []
        }

    t0 = time.time()
    speech_emotion, speech_conf, raw_emotions = get_ser_from_api(audio, SAMPLE_RATE)
    print(f"[Timing] SER took {time.time()-t0:.2f} seconds")

    t0 = time.time()
    transcript = run_stt(audio)
    print(f"[Timing] STT took {time.time()-t0:.2f} seconds")

    return {
        "transcript": transcript,
        "speech_emotion": speech_emotion,
        "speech_conf": speech_conf,
        "raw_emotions": raw_emotions
    }
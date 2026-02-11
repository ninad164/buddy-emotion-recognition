import os
from dotenv import load_dotenv

load_dotenv()

FUSED_EMOTIONS = ["neutral", "happy", "sad", "surprise", "anger"]
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 5   # seconds
EMOTION_BLOB_PATH = r"d:/thesis/models/face-detection-retail-0004.blob" # path to your OAK-D emotion recognition blob
LANGUAGE = "en"
BEAM_SIZE = 1
COMPUTE_TYPE = "float16"
MODE = "fusion"   # options: "voice" or "fusion"

# Local endpoints
LOCAL_LLM_URL = "http://localhost:8000/chat"   # your local LLM server
SER_API_URL   = "http://localhost:8001/ser"

LLM_MODEL_PATH = os.getenv(
    "LLM_MODEL_PATH",
    r"D:\thesis\Meta-Llama-3.1-8B-Instruct-GGUF\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
)

ENABLE_TTS = True  # or False if you want silent mode
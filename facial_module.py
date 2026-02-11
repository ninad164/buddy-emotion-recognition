# facial_module.py
import depthai as dai
from config import EMOTION_BLOB_PATH, FUSED_EMOTIONS

# Labels for the OAK-D emotion recognition blob
FER_LABELS = ["neutral", "happy", "sad", "surprise", "anger"]

def run_facial_emotion_model():
    """
    Run the OAK-D pipeline with the emotion_recognition.blob
    and return (emotion_label, confidence).
    """
    # Create pipeline
    pipeline = dai.Pipeline()

    # Define sources and outputs
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(300,300)         # should match the size in the blob
    cam.setInterleaved(False)
    cam.setFps(30)

    # Neural network node
    nn = pipeline.createNeuralNetwork()
    nn.setBlobPath(EMOTION_BLOB_PATH)
    cam.preview.link(nn.input)

    # XLink output
    xout = pipeline.createXLinkOut()
    xout.setStreamName("nn")
    nn.out.link(xout.input)

    # Run pipeline
    with dai.Device(pipeline) as device:
        q_nn = device.getOutputQueue(name="nn", maxSize=1, blocking=True)
        in_nn = q_nn.get()  # blocking call, waits for one inference
        scores = in_nn.getFirstLayerFp16()
        if not scores:
            print("[FER] Empty NN output, defaulting to neutral")
            return "neutral", 0.0

        # Pick max score
        idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        emotion = FER_LABELS[idx]
        conf = scores[idx]

        return emotion, conf


def process_face():
    """
    Capture one inference from OAK-D and return dict with emotion + confidence.
    """
    try:
        emotion, conf = run_facial_emotion_model()
        if emotion not in FUSED_EMOTIONS:
            emotion = "neutral"
        print(f"[Facial Emotion] {emotion} ({conf:.2f})")
        return {
            "transcript": None,
            "face_emotion": emotion,
            "face_conf": conf
        }
    except Exception as e:
        print(f"[FER Error] {e}")
        return {"transcript": None, "face_emotion": "neutral", "face_conf": 0.0}
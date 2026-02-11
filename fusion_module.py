# fusion_module.py
import threading
import time
import numpy as np
import depthai as dai
import sounddevice as sd
from voice_module import run_stt, run_ser
from config import DURATION, SAMPLE_RATE, CHANNELS

# Weights and labels
VOICE_WEIGHT = 0.35
FACE_WEIGHT = 0.65
LABELS = ["neutral", "happy", "sad", "surprise", "anger"]

def capture_audio():
    """Record audio for fusion window without extra Enter prompt."""
    print("🎙️ Recording...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=CHANNELS,
                   dtype='float32')
    sd.wait()
    print("✅ Recording complete.")
    return np.squeeze(audio)

def process_face(duration: float = DURATION) -> str:
    """
    Start Oak‑D pipeline when called and run emotion inference for `duration` seconds.
    Building the pipeline inside ensures the camera starts only in the fused window.
    """
    emotions = []
    start = time.time()

    # Build pipeline locally so it starts only when this function is invoked
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(300, 300)
    cam.setInterleaved(False)
    cam.setFps(10)

    # Face detection NN
    face_det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    face_det_nn.setBlobPath("d:/thesis/models/face-detection-retail-0004.blob")
    face_det_nn.setConfidenceThreshold(0.5)
    cam.preview.link(face_det_nn.input)

    # Script node to parse detections and send crop configs
    script = pipeline.create(dai.node.Script)
    face_det_nn.out.link(script.inputs["nn_in"])

    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(64, 64)
    manip.initialConfig.setKeepAspectRatio(False)

    cam.preview.link(manip.inputImage)
    script.outputs["manip_cfg"].link(manip.inputConfig)

    # Emotion recognition NN
    emotion_nn = pipeline.create(dai.node.NeuralNetwork)
    emotion_nn.setBlobPath(
        "d:/thesis/models/emotions-recognition-retail-0003_openvino_2022.1_6shave.blob"
    )
    manip.out.link(emotion_nn.input)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("emotions")
    emotion_nn.out.link(xout.input)

    # Script node code (runs on device)
    script.setScript(
        """
        while True:
            det_in = node.io['nn_in'].get()
            if not det_in.detections: continue
            for det in det_in.detections:
                xmin = max(0.0, det.xmin)
                ymin = max(0.0, det.ymin)
                xmax = min(1.0, det.xmax)
                ymax = min(1.0, det.ymax)
                cfg = ImageManipConfig()
                cfg.setCropRect(xmin, ymin, xmax, ymax)
                cfg.setResize(64, 64)
                cfg.setKeepAspectRatio(False)
                node.io['manip_cfg'].send(cfg)
        """
    )

    # Open and close device inside this function
    try:
        with dai.Device(pipeline) as device:
            q_emotions = device.getOutputQueue(name="emotions", maxSize=1, blocking=False)

            while time.time() - start < duration:
                in_nn = q_emotions.tryGet()
                if in_nn is None:
                    continue
                scores = in_nn.getFirstLayerFp16()
                if not scores:
                    continue
                idx = int(np.argmax(scores))
                emotions.append(LABELS[idx])

    except Exception:
        # If device fails to open, fall back to neutral for this window
        return "neutral"

    if emotions:
        # Mode over the window
        return max(set(emotions), key=emotions.count)

    return "neutral"

def run_fusion():
    transcript = [None]
    voice_emotion = [("neutral", 1.0)]
    face_emotion = ["neutral"]

    # Shared start signal
    start_event = threading.Event()

    def audio_task():
        start_event.wait()  # wait until Enter is pressed
        audio = capture_audio()  # records for DURATION
        if audio is None:
            return
        transcript[0] = run_stt(audio)
        voice_emotion[0] = run_ser(audio)

    def face_task():
        start_event.wait()  # wait until Enter is pressed
        face_emotion[0] = process_face(duration=DURATION)

    # Launch threads
    t_audio = threading.Thread(target=audio_task, daemon=True)
    t_face = threading.Thread(target=face_task, daemon=True)
    t_audio.start()
    t_face.start()

    # Gate start on Enter
    input("\n🎤 Press Enter to start recording...")
    start_event.set()

    # Wait for both tasks
    t_audio.join()
    t_face.join()

    # Fuse emotions
    v_label, v_conf = voice_emotion[0]
    f_label = face_emotion[0]

    v_vec = np.zeros(len(LABELS))
    f_vec = np.zeros(len(LABELS))
    if v_label in LABELS:
        v_vec[LABELS.index(v_label)] = v_conf
    if f_label in LABELS:
        f_vec[LABELS.index(f_label)] = 1.0

    fused_vec = VOICE_WEIGHT * v_vec + FACE_WEIGHT * f_vec
    final_emotion = LABELS[int(np.argmax(fused_vec))]

    return {
        "transcript": transcript[0],
        "fusion": {
            "voice_emotion": v_label,
            "voice_confidence": v_conf,
            "face_emotion": f_label,
            "final_emotion": final_emotion,
        },
    }

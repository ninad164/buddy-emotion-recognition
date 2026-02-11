import depthai as dai, time, numpy as np

LABELS = ["neutral","happy","sad","surprise","anger","fear","disgust"]

pipeline = dai.Pipeline()
cam = pipeline.createColorCamera()
cam.setPreviewSize(300, 300)
cam.setInterleaved(False)
cam.setFps(10)

face_det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
face_det_nn.setBlobPath("d:/thesis/models/face-detection-retail-0004.blob")
face_det_nn.setConfidenceThreshold(0.5)
cam.preview.link(face_det_nn.input)

script = pipeline.create(dai.node.Script)
face_det_nn.out.link(script.inputs['nn_in'])

manip = pipeline.createImageManip()
manip.setResize(64, 64)
manip.setKeepAspectRatio(False)
cam.preview.link(manip.inputImage)
script.outputs['manip_cfg'].link(manip.inputConfig)

emotion_nn = pipeline.createNeuralNetwork()
emotion_nn.setBlobPath("d:/thesis/models/emotions-recognition-retail-0003.blob")
manip.out.link(emotion_nn.input)

xout = pipeline.createXLinkOut()
xout.setStreamName("emotions")
emotion_nn.out.link(xout.input)

script.setScript("""
while True:
    det_in = node.io['nn_in'].get()
    if not det_in.detections: continue
    for det in det_in.detections:
        cfg = dai.ImageManipConfig()
        cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax)
        node.io['manip_cfg'].send(cfg)
""")

with dai.Device(pipeline) as device:
    q = device.getOutputQueue("emotions", maxSize=1, blocking=False)
    while True:
        in_nn = q.tryGet()
        if in_nn is not None:
            scores = in_nn.getFirstLayerFp16()
            if scores:
                idx = int(np.argmax(scores))
                print("Emotion:", LABELS[idx])
        time.sleep(0.05)
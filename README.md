# Buddy Emotion Recognition

A real-time multimodal emotion recognition system for human-robot interaction, combining facial expression and speech-based analysis.

---

## 🚀 Overview

This project implements a perception pipeline that enables robots to infer human emotional state using both visual and auditory inputs.

It is designed as a core component for:
- Companion robots
- Social robotics systems
- Human-centered interaction frameworks

The system processes live camera and microphone input to classify emotions in real time.

---

## 🧠 System Design

The pipeline consists of:

### 1. Visual Emotion Recognition
- Face detection and preprocessing
- Emotion classification from facial expressions
- Real-time inference from camera stream

### 2. Audio Emotion Recognition
- Speech signal acquisition
- Feature extraction (e.g., spectrogram / embeddings)
- Emotion classification from voice

### 3. Multimodal Fusion
- Combines predictions from vision and audio
- Improves robustness over single-modality systems

---

## ⚙️ Tech Stack

- Python
- OpenCV
- NumPy
- Deep Learning models (vision + audio)
- OAK-D Camera (if used)
- ReSpeaker Mic Array (if used)

---

## ▶️ Running the Project

```bash
git clone https://github.com/ninad164/buddy-emotion-recognition.git
cd buddy-emotion-recognition
pip install -r requirements.txt
````

Run the main pipeline:

```bash
python main.py
```

> Replace `main.py` if your entry file is different

---

## 📊 Key Features

* Real-time emotion recognition
* Multimodal perception (vision + audio)
* Modular pipeline design
* Suitable for integration with robotic systems

---

## 🔬 Use Case

This system is designed for integration into a robotic platform where:

* the robot perceives human emotional state
* adapts behavior accordingly
* improves trust and interaction quality

---

## 📌 Future Improvements

* Improve fusion strategy (late vs early fusion)
* Optimize for embedded deployment (Jetson)
* Expand emotion classes
* Improve robustness in noisy environments

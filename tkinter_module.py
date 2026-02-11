# tkinter_module.py
import tkinter as tk
import time, threading
import pyttsx3

engine = pyttsx3.init()

class TalkingFace:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("LLM Talking Face")
        self.root.state("zoomed")
        self.canvas = tk.Canvas(self.root, bg="white")
        self.canvas.pack(expand=True, fill="both")

        self.current_emotion = "neutral"
        self.mouth_open = False

        # Redraw whenever window resizes
        self.canvas.bind("<Configure>", lambda e: self.draw_face(self.current_emotion, self.mouth_open))

    def draw_face(self, emotion="neutral", mouth_open=False):
        self.canvas.delete("all")
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        size = min(w, h) * 0.8
        cx, cy = w // 2, h // 2
        r = size // 2

        # Head
        self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill="#fce5cd", outline="black")

        # Eyes
        eye_offset_x, eye_offset_y = r*0.35, r*0.3
        eye_size = r*0.1
        lx, ly = cx-eye_offset_x, cy-eye_offset_y
        rx, ry = cx+eye_offset_x, cy-eye_offset_y

        if emotion == "happy":
            self.canvas.create_arc(lx-eye_size, ly-eye_size, lx+eye_size, ly+eye_size,
                                   start=0, extent=180, style=tk.ARC, width=3)
            self.canvas.create_arc(rx-eye_size, ry-eye_size, rx+eye_size, ry+eye_size,
                                   start=0, extent=180, style=tk.ARC, width=3)
        elif emotion == "surprised":
            self.canvas.create_oval(lx-eye_size, ly-eye_size, lx+eye_size, ly+eye_size,
                                    fill="white", outline="black", width=3)
            self.canvas.create_oval(rx-eye_size, ry-eye_size, rx+eye_size, ry+eye_size,
                                    fill="white", outline="black", width=3)
            pupil = eye_size * 0.4
            self.canvas.create_oval(lx-pupil, ly-pupil, lx+pupil, ly+pupil, fill="black")
            self.canvas.create_oval(rx-pupil, ry-pupil, rx+pupil, ry+pupil, fill="black")
        else:
            self.canvas.create_oval(lx-eye_size, ly-eye_size, lx+eye_size, ly+eye_size, fill="black")
            self.canvas.create_oval(rx-eye_size, ry-eye_size, rx+eye_size, ry+eye_size, fill="black")

        # Eyebrows
        brow_offset_y = r*0.15
        if emotion == "angry":
            self.canvas.create_line(lx-eye_size, ly-brow_offset_y, lx+eye_size, ly-brow_offset_y/2, width=4)
            self.canvas.create_line(rx-eye_size, ry-brow_offset_y/2, rx+eye_size, ry-brow_offset_y, width=4)
        elif emotion == "sad":
            self.canvas.create_line(lx-eye_size, ly-brow_offset_y/2, lx+eye_size, ly-brow_offset_y, width=3)
            self.canvas.create_line(rx-eye_size, ry-brow_offset_y, rx+eye_size, ry-brow_offset_y/2, width=3)
        elif emotion == "surprised":
            self.canvas.create_line(lx-eye_size, ly-brow_offset_y*1.5, lx+eye_size, ly-brow_offset_y*1.5, width=3)
            self.canvas.create_line(rx-eye_size, ry-brow_offset_y*1.5, rx+eye_size, ry-brow_offset_y*1.5, width=3)

        # Mouth
        if mouth_open:
            self.canvas.create_oval(cx-r*0.2, cy+r*0.2, cx+r*0.2, cy+r*0.45, fill="black")
        else:
            self.canvas.create_line(cx-r*0.25, cy+r*0.35, cx+r*0.25, cy+r*0.35, width=4)

        self.root.update()

    def update_emotion(self, emotion):
        """Update face instantly to a new emotion (no speaking)."""
        self.current_emotion = emotion
        self.draw_face(emotion, self.mouth_open)

    def speak(self, text, emotion="neutral"):
        """Blocking mouth animation until text is finished."""
        self.current_emotion = emotion
        words = text.split()
        for word in words:
            # Open mouth
            self.mouth_open = True
            self.draw_face(emotion, mouth_open=True)
            time.sleep(0.2)

            # Close mouth
            self.mouth_open = False
            self.draw_face(emotion, mouth_open=False)
            time.sleep(0.2)

        # End with mouth closed
        self.mouth_open = False
        self.draw_face(emotion, mouth_open=False)

        def run(self):
            self.root.mainloop()


# Example standalone run
if __name__ == "__main__":
    face = TalkingFace()

    def demo():
        # Simulate LLM responses
        responses = [
            ("Hello, I am your LLM avatar", "happy"),
            ("Sometimes I feel sad", "sad"),
            ("That makes me angry", "angry"),
            ("Wow, I am surprised", "happy"),
            ("Back to neutral now", "neutral"),
        ]
        for text, emo in responses:
            face.speak(text, emotion=emo)
            time.sleep(1)

    threading.Thread(target=demo, daemon=True).start()
    face.run()
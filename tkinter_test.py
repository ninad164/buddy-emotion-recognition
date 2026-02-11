# tkinter_test.py
import tkinter as tk
import time, threading

class TalkingFace:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Talking Face")
        self.root.state("zoomed")  # start maximized
        self.canvas = tk.Canvas(self.root, bg="white")
        self.canvas.pack(expand=True, fill="both")

        self.current_emotion = "neutral"
        self.mouth_open = False

        # Redraw whenever window resizes
        self.canvas.bind("<Configure>", lambda e: self.draw_face(self.current_emotion, self.mouth_open))

    def draw_face(self, emotion="neutral", mouth_open=False):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        size = min(w, h) * 0.8
        cx, cy = w // 2, h // 2
        r = size // 2

        # Head
        self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill="#fce5cd", outline="black")

        # Eyes
        eye_offset_x, eye_offset_y = r*0.35, r*0.3
        eye_size = r*0.1
        left_eye_x, left_eye_y = cx-eye_offset_x, cy-eye_offset_y
        right_eye_x, right_eye_y = cx+eye_offset_x, cy-eye_offset_y

        if emotion == "happy":
            # smiling eyes as arcs (no eyebrows)
            self.canvas.create_arc(left_eye_x-eye_size, left_eye_y-eye_size,
                                   left_eye_x+eye_size, left_eye_y+eye_size,
                                   start=0, extent=180, style=tk.ARC, width=3)
            self.canvas.create_arc(right_eye_x-eye_size, right_eye_y-eye_size,
                                   right_eye_x+eye_size, right_eye_y+eye_size,
                                   start=0, extent=180, style=tk.ARC, width=3)

        elif emotion == "surprised":
            # white eyes with black pupils
            self.canvas.create_oval(left_eye_x-eye_size, left_eye_y-eye_size,
                                    left_eye_x+eye_size, left_eye_y+eye_size,
                                    fill="black", outline="black", width=3)
            self.canvas.create_oval(right_eye_x-eye_size, right_eye_y-eye_size,
                                    right_eye_x+eye_size, right_eye_y+eye_size,
                                    fill="black", outline="black", width=3)

        else:
            # normal round eyes
            self.canvas.create_oval(left_eye_x-eye_size, left_eye_y-eye_size,
                                    left_eye_x+eye_size, left_eye_y+eye_size, fill="black")
            self.canvas.create_oval(right_eye_x-eye_size, right_eye_y-eye_size,
                                    right_eye_x+eye_size, right_eye_y+eye_size, fill="black")

        # Eyebrows (skip entirely if happy or neutral)
        brow_offset_y = r*0.15
        if emotion == "angry":
            self.canvas.create_line(left_eye_x-eye_size, left_eye_y-brow_offset_y,
                                    left_eye_x+eye_size, left_eye_y-brow_offset_y/2, width=4)
            self.canvas.create_line(right_eye_x-eye_size, right_eye_y-brow_offset_y/2,
                                    right_eye_x+eye_size, right_eye_y-brow_offset_y, width=4)
        elif emotion == "sad":
            self.canvas.create_line(left_eye_x-eye_size, left_eye_y-brow_offset_y/2,
                                    left_eye_x+eye_size, left_eye_y-brow_offset_y, width=3)
            self.canvas.create_line(right_eye_x-eye_size, right_eye_y-brow_offset_y,
                                    right_eye_x+eye_size, right_eye_y-brow_offset_y/2, width=3)
        elif emotion == "surprised":
            self.canvas.create_line(left_eye_x-eye_size, left_eye_y-brow_offset_y*1.5,
                                    left_eye_x+eye_size, left_eye_y-brow_offset_y*1.5, width=3)
            self.canvas.create_line(right_eye_x-eye_size, right_eye_y-brow_offset_y*1.5,
                                    right_eye_x+eye_size, right_eye_y-brow_offset_y*1.5, width=3)

        # Mouth (always just open/close)
        if mouth_open:
            self.canvas.create_oval(cx-r*0.2, cy+r*0.2, cx+r*0.2, cy+r*0.45, fill="black")
        else:
            self.canvas.create_line(cx-r*0.25, cy+r*0.35, cx+r*0.25, cy+r*0.35, width=4)

        self.root.update()

    def speak(self, sentence, emotion="neutral"):
        """Animate mouth open/close while 'speaking' a sentence. Eyes show emotion."""
        self.current_emotion = emotion
        words = sentence.split()
        for word in words:
            self.mouth_open = True
            self.draw_face(emotion, mouth_open=True)
            time.sleep(0.2)
            self.mouth_open = False
            self.draw_face(emotion, mouth_open=False)
            time.sleep(0.2)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    face = TalkingFace()

    def demo():
        face.speak("Hello there, I am happy to see you", emotion="happy")
        time.sleep(1)
        face.speak("I am feeling sad today", emotion="sad")
        time.sleep(1)
        face.speak("That makes me angry", emotion="angry")
        time.sleep(1)
        face.speak("Wow, I am surprised", emotion="surprised")
        time.sleep(1)
        face.speak("Back to neutral now", emotion="neutral")

    threading.Thread(target=demo, daemon=True).start()
    face.run()
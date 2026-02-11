import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from fusion_module import run_fusion
from datetime import datetime
from llm_module import generate_response
from tkinter_module import TalkingFace
import threading
import time

MODE="voice"  # "voice" or "fusion"

def main():
    print("🎤 Assistant ready in FUSION mode with Avatar.\n")

    # # Create the TalkingFace avatar
    # face = TalkingFace()

    def loop():
        try:
            while True:
                result = run_fusion()

                # Timestamp
                print("Timestamp:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

                # Print results
                print("SER:", result["fusion"]["voice_emotion"])
                print("FER:", result["fusion"]["face_emotion"])
                print("Transcript:", result["transcript"])
                print("Final emotion:", result["fusion"]["final_emotion"])
                print("-" * 60)

                # Build empathetic LLM response
                transcript = result["transcript"]
                final_emotion = result["fusion"]["final_emotion"]

                llm_reply = generate_response(transcript, emotion=final_emotion)

                print("LLM response:", llm_reply)
                print("=" * 60)

                # Animate the face speaking the LLM reply
                # face.speak(llm_reply, emotion=final_emotion)

                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\nExiting...")

    # Run the fusion/LLM loop in a background thread
    threading.Thread(target=loop, daemon=True).start()

    # Start Tkinter mainloop (blocking)
    face.run()

from script_runner import run_voice_only_script

if __name__ == "__main__":
    if MODE == "voice":   # voice mode
        from script_runner import run_voice_only_script
        run_voice_only_script()
    elif MODE == "fusion":   # fusion mode
        from script_face import run_face_script
        run_face_script()
        
# if __name__ == "__main__":
#     main()
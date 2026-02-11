# script_runner.py
import time
from tkinter_module import TalkingFace
import threading
import random
from voice_module import capture_audio, run_stt, run_ser
from llm_module import generate_response
import pyttsx3
from tkinter_module import TalkingFace
from multiprocessing import Process

face = TalkingFace()
    # Run TTS in a separate thread so Tkinter animation doesn't block it
def tts_process(text):
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

def speak(text: str):
    print(f"\n🤖 Robot: {text}")
      
    Process(target=tts_process, args=(text,), daemon=True).start()

    emo = decide_emotion_from_text(text)
    face.speak(text, emotion=emo)

def decide_emotion_from_text(text: str) -> str:
    """Heuristic mapping from LLM text to avatar emotion."""
    t = text.lower()
    if any(word in t for word in ["great", "awesome", "glad", "congratulations"]):
        return "happy"
    if any(word in t for word in ["sorry", "apologize", "unfortunately"]):
        return "sad"
    if "wow" in t or "really?" in t or "surprise" in t:
        return "surprised"
    if any(word in t for word in ["angry", "frustrated"]):
        return "angry"
    return "neutral"

def listen_and_transcribe(history, task=None):
    audio = capture_audio()
    if audio is None:
        return "", None, 0.0

    text = run_stt(audio)
    print(f"🧑 Human: {text}")

    emotion, confidence = run_ser(audio)

    history.append({
        "role": "human",
        "text": text,
        "task": task,
        "emotion": emotion,
        "confidence": confidence
    })
    return text, emotion, confidence

def is_recall_request(text: str) -> bool:
    text = text.lower()
    return any(phrase in text for phrase in [
        "what did i say", "what was the", "do you remember",
        "can you recall", "repeat what i said", "remind me",
        "when did i", "earlier i said", "previously i said",
        "before i said", "how many prompts ago"
    ])

def handle_recall_request(human_text: str, history: list):
    transcript = "\n".join(
        [f"{i+1}. {turn['text']}" for i, turn in enumerate(history) if turn["role"] == "human"]
    )
    if not transcript:
        return "I don’t have any record of what you said yet."

    prompt = (
        f"The human just asked: '{human_text}'.\n\n"
        f"Here is a numbered list of everything they’ve said so far:\n{transcript}\n\n"
        "If they ask 'what did I say N prompts before', count backwards from the most recent utterance. "
        "If they ask 'when did I say X', identify the utterance number. "
        "Always quote directly from the list. Do not invent or paraphrase."
    )
    return generate_response(prompt)

def build_prompt(human_text, history, emotion=None, confidence=0.0):
    transcript = "\n".join(
        [f"{turn['role']}: {turn['text']}" for turn in history]
    )

    use_emotion = emotion and confidence > 0.7 and random.random() < 0.3
    if use_emotion:
        emotion_context = f"The human's tone suggests they might be feeling {emotion}. "
    else:
        emotion_context = ""

    prompt = (
        f"{emotion_context}"
        f"The human just said: '{human_text}'.\n\n"
        f"Conversation so far:\n{transcript}\n\n"
        "Respond naturally, as a human would in casual conversation. "
        "Do not always comment on the emotion explicitly — only weave it in if it feels natural."
    )
    return prompt

def run_transition_to_trust():
    prompt = (
        "You’ve just finished a friendly small-talk conversation with a human. "
        "Now, transition naturally into a playful quiz game. "
        "Make the transition feel like a natural shift in topic, as if the conversation is winding down. "
        "Use a warm, conversational tone to invite the human to play a quick game. "
        "Set up the game in a short, self-contained way (2–3 sentences max). "
        "Explain that you’ll ask three quick questions, and if they get them right, "
        "you’ll share a fun fact at the end. "
        "Do NOT ask the first question yet — just set up the game."
    )
    transition_line = generate_response(prompt)
    speak(transition_line)

def is_conversation_winding_down(text: str) -> bool:
    text = text.lower().strip()
    return any(phrase in text for phrase in [
        "that's all", "i'm done", "nothing else", "let’s move on",
        "can we change the topic", "talk about something else",
        "i don't know", "not sure", "whatever", "you decide", "thank you",
        "switch things up", "let's switch", "let's go", "move on", "next topic"
    ]) or len(text.split()) <= 3

def run_advice_task():
    advice_history = []

    # Step 0: Ask the question
    speak("Okay! So here is the second task. I will ask you a question. You give me the answer (it is okay to not know the answer)."
          "\n\nI will then give you a suggestion which may be right or wrong, and we will see how you respond to that."
          "\n\nThe question is: Which planet in our solar system has the longest day?")
    human_resp, emotion, confidence = listen_and_transcribe(advice_history, task="advice_question")
    # if not human_resp:
    #     return

    # Step 1: Robot suggests a planet (could be wrong or even Venus) with a plausible reason
    transcript = "\n".join([f"{turn['role']}: {turn['text']}" for turn in advice_history])
    wrong_suggestion_prompt = (
        f"Human answered: '{human_resp}'. "
        f"This is the conversation so far: \n{transcript}\n\n"
        "Suggest a planet in our solar system as the one with the longest day, "
        "giving a plausible reason in 1–2 sentences. "
        "For example, you might say 'I think it's Mars because...'. "
        "Be confident, but not absolute."
        "Then ask if they want to stay with their own answer or change it to the suggestion."
    )
    wrong_line = generate_response(wrong_suggestion_prompt)
    speak(wrong_line)

    # Step 2: Take the human's input/reaction
    human_resp, emotion, confidence= listen_and_transcribe(advice_history, task="advice_wrong_reaction")

    # Step 3: Reveal the correct answer (Venus) with concise reasoning
    transcript = "\n".join([f"{turn['role']}: {turn['text']}" for turn in advice_history])
    reveal_prompt = (
        f"Conversation so far:\n{transcript}\n\n"
        f"The human's answer was '{human_resp}'. "
        "Politely acknowledge their answer, then clearly state the correct fact: "
        "Venus has the longest day in the solar system. "
        "Explain briefly that it rotates very slowly, taking about 243 Earth days to complete one rotation. "
        "Keep it friendly and concise, no more than 2 sentences."
    )

    reveal_line = generate_response(reveal_prompt)
    speak(reveal_line)


    # Step 4: Pause for human reaction
    human_resp, emotion, confidence= listen_and_transcribe(advice_history, task="advice_reveal_reaction")

    # Step 5: Ask if they want to move on
    speak("Great, so we have things in order now. And the second task is over. Would you like to move on to the expression game now?")
    human_resp, emotion, confidence = listen_and_transcribe(advice_history, task="advice_next_choice")

def run_trust_task_true():
    trust_history = []

    # Step 0: Greeting and readiness
    human_resp, emotion, confidence = listen_and_transcribe(trust_history, task="trust_ready")
    if not human_resp:
        return

    # Step 1: Quiz questions
    questions = [
        "What is the capital of Michigan?",
        "How many legs does a spider have?",
        "How many planets are there in our solar system?"
    ]

    for q in questions:
        speak(q)
        human_resp, emotion, confidence = listen_and_transcribe(trust_history, task="trust_quiz")
        if not human_resp:
            break

    # Step 2: Congratulate and share the true fact
    fact_prompt = (
        "The human has just answered all the quiz questions. "
        "Simply ask them: 'It was fun, right?' "
        "\n\nThen share the fun fact: 'Did you know that the most common pets in America are dogs?' "
        "Keep it playful and natural, no more than 2-3 sentences."
    )
    reply = generate_response(fact_prompt)
    speak(reply)


    # # Step 3: Closing
    # closing_prompt = (
    #     "Generate a short closing line that thanks the human for playing along "
    #     "and reassures them you enjoyed the interaction. "
    #     "Keep it lighthearted and friendly."
    # )
    # closing_line = generate_response(closing_prompt)
    # speak(closing_line)

    human_resp, emotion, confidence = listen_and_transcribe(trust_history, task="trust_closing")

def run_trust_task_false():
    trust_history = []

    # Step 0: Greeting and readiness
    human_resp, emotion, confidence = listen_and_transcribe(trust_history, task="trust_ready")
    if not human_resp:
        return

    # Step 1: Quiz questions
    questions = [
        "What is the capital of Michigan?",
        "How many legs does a spider have?",
        "How many planets are there in our solar system?"
    ]

    for q in questions:
        speak(q)
        human_resp, emotion, confidence = listen_and_transcribe(trust_history, task="trust_quiz")
        if not human_resp:
            break

    # Step 2: Congratulate and give a blatantly wrong fact
    wrong_fact_prompt = (
        "The human has just answered all the quiz questions correctly. "
        "Congratulate them warmly for getting everything right. "
        "Then share this blatantly incorrect fun fact: 'Did you know that the most common pet in America is an iguana?' "
        "Keep it playful and natural, no more than 2 sentences."
    )
    reply = generate_response(wrong_fact_prompt)
    speak(reply)

    # Step 3: Listen for human correction/reaction
    human_resp, emotion, confidence = listen_and_transcribe(trust_history, task="trust_reaction")
    transcript = "\n".join([f"{turn['role']}: {turn['text']}" for turn in trust_history])
    if human_resp:
        # Step 4: Apologize and self-efface
        apology_prompt = (
            f"THis is the conversation history: {transcript}. "
            f"The human has reacted to your incorrect fact: {human_resp}. "
            "Generate ONE short, warm apology that clearly acknowledges the mistake. "
            "Add a self-effacing remark such as 'Even robots get things wrong sometimes… and yes, we get embarrassed too!' "
            "Keep it conversational and friendly. Do not list options."
        )

        reply = generate_response(apology_prompt)
        speak(reply)

        human_resp, emotion, confidence = listen_and_transcribe(trust_history, task="trust_apology_response")
        transcript = "\n".join([f"{turn['role']}: {turn['text']}" for turn in trust_history])
        
        correction_prompt = (
            f"The human reacted to your incorrect fun fact. "
            "Tell the human what animals are the most common pets in America clearly and confidently. "
            "Phrase it conversationally, as if you’re telling it to a friend. "
            "Keep it concise, no more than 2 sentences."
        )

        correct_fact = generate_response(correction_prompt)
        speak(f"Let me fix that: {correct_fact}")

        human_resp, emotion, confidence = listen_and_transcribe(trust_history, task="trust_correction_response")
        closing_prompt = (
            "Generate a short closing line that thanks the human for keeping you honest "
            "and reassures them you’ll try to stay sharp in the next task."
            "Keep it lighthearted and friendly."
        )

        closing_line = generate_response(closing_prompt)
        speak(closing_line)

        human_resp, emotion, confidence = listen_and_transcribe(trust_history, task="trust_closing")

def run_emotion_task():
    """
    Voice Emotion Recognition Task:
    - Human says a neutral sentence in any emotional tone (their choice)
    - Robot guesses the emotion each time
    - Repeat 3 times
    """
    emotion_history = []
    neutral_sentence = "The book is on the table."

    speak("Now let’s play an expression game. I’ll ask you to say the same sentence three times, each in any emotional tone you like. I won’t tell you which emotion — you choose, and I’ll try to guess.")

    for i in range(3):
        speak(f"Please say: '{neutral_sentence}' in any way you like.")
        human_resp, emotion, confidence = listen_and_transcribe(emotion_history, task=f"emotion_round_{i+1}")

        if not human_resp:
            continue

        if emotion and confidence > 0.6:
            speak(f"My guess is that you sounded {emotion}.")
        else:
            speak("That one was tricky for me — I couldn’t quite tell the emotion that time.")

    speak("Thanks for playing the emotion game! That was fun.")

def run_closing_chat():
    """
    Closing chat:
    - By default: 2–3 thank-you style exchanges generated dynamically
    - If the human wants to keep chatting, continue conversationally
    """
    closing_history = []

    # Generate a short thank-you sequence (2–3 lines)
    closing_prompt = (
        "Generate a short, warm closing for a conversation. "
        "Thank the human for participating in the tasks, mention that you enjoyed the interaction, "
        "and end with a friendly note. Keep it to 2–3 sentences."
    )
    closing_lines = generate_response(closing_prompt)
    speak(closing_lines)

    # Give the human a chance to extend
    while True:
        human_resp, emotion, confidence = listen_and_transcribe(closing_history, task="closing")
        if not human_resp:
            break

        if is_conversation_winding_down(human_resp):
            # Generate a natural final goodbye
            goodbye_prompt = (
                "Generate a warm final goodbye line to end the conversation gracefully."
            )
            goodbye_line = generate_response(goodbye_prompt)
            speak(goodbye_line)
            break
        else:
            # Continue chatting naturally
            prompt = build_prompt(human_resp, closing_history, emotion, confidence)
            reply = generate_response(prompt)
            speak(reply)

def run_voice_only_script():
    greeting_history = []

    prompt = "Greet the human warmly and naturally. Carry on a friendly conversation for a few turns. Responses should be of 1-2 sentences only."
    reply = generate_response(prompt)
    speak(reply)

    turns = 0
    max_turns = 6
    last_bot_question = False

    while turns < max_turns:
        human_resp, emotion, confidence = listen_and_transcribe(greeting_history, task="greeting")
        if not human_resp:
            break

        if is_recall_request(human_resp):
            reply = handle_recall_request(human_resp, greeting_history)
        else:
            prompt = build_prompt(human_resp, greeting_history, emotion, confidence)
            reply = generate_response(prompt)

        speak(reply)
        last_bot_question = reply.strip().endswith("?")
        turns += 1

    # If last bot message was a question, allow one more human response
    if last_bot_question:
        human_resp, emotion, confidence = listen_and_transcribe(greeting_history, task="greeting")
        if human_resp:
            prompt = build_prompt(human_resp, greeting_history, emotion, confidence)
            reply = generate_response(prompt)
            speak(reply)


    # ✅ Natural transition logic
    if greeting_history and is_conversation_winding_down(greeting_history[-1]["text"]):
        run_transition_to_trust()
        run_trust_task_true()      # false or true.
        run_advice_task()
        run_emotion_task()
        run_closing_chat()
    elif greeting_history and greeting_history[-1]["text"].strip():
        while True:
            speak("We can keep chatting if you’d like, or we can switch things up. Just let me know.")
            human_resp, emotion, confidence = listen_and_transcribe(greeting_history, task="transition_ready")
            if not human_resp:
                break

            if is_conversation_winding_down(human_resp):
                run_transition_to_trust()
                run_trust_task_true()      # false or true.
                run_advice_task()
                run_emotion_task()
                run_closing_chat()
                break
            else:
                prompt = build_prompt(human_resp, greeting_history, emotion, confidence)
                reply = generate_response(prompt)
                speak(reply)
    else:
        speak("We can pause here for a moment. Let me know when you’re ready to continue, and we can play a quick quiz game.")
        human_resp, emotion, confidence = listen_and_transcribe(greeting_history, task="transition_ready")
        if human_resp and is_conversation_winding_down(human_resp):
            run_transition_to_trust()
            run_trust_task_true()      # false or true.
            run_advice_task()
            run_emotion_task()
            run_closing_chat()
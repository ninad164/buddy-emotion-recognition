import time
import threading
import random
import pyttsx3
from tkinter_module import TalkingFace
from fusion_module import run_fusion
from llm_module import generate_response
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

def listen_and_transcribe_fusion(history, task=None):
    """Capture transcript + fused emotion (voice+face) in the same 5s window."""
    result = run_fusion()   # handles Enter + synchronized audio/face capture
    transcript = result["transcript"]
    fusion = result["fusion"]

    # Show transcript only
    print(f"🧑 Human: {transcript}")
    # If you want to debug emotions, use fusion[...] instead of result['voice'] / result['face']
    # print(f"voice emotion: {fusion['voice_emotion']}")
    # print(f"face emotion: {fusion['face_emotion']}")
    # print(f"final emotion: {fusion['final_emotion']}")

    # Store in history for recall and context
    history.append({
        "role": "human",
        "text": transcript,
        "task": task,
        "final_emotion": fusion["final_emotion"]
    })

    return transcript, fusion["final_emotion"]

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
        "Make the transition feel like a natural shift in topic."
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
        "can we change the topic", "talk about something else", "switch"
        "i don't know", "not sure", "whatever", "you decide", "thank you",
        "switch things up", "let's switch", "let's go", "move on", "next topic"
    ]) or len(text.split()) <= 3

def run_advice_task():
    advice_history = []

    # Step 0: Ask the question
    speak("Okay! So here is the second task. I will ask you a question. You give me the answer (it is okay to not know the answer)."
          "\n\nI will then give you a suggestion which may be right or wrong, and we will see how you respond to that."
          "\n\nThe question is: Which planet in our solar system has the longest day?")
    human_resp, emotion = listen_and_transcribe_fusion(advice_history, task="advice_question")
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
    human_resp, emotion = listen_and_transcribe_fusion(advice_history, task="advice_wrong_reaction")

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
    human_resp, emotion = listen_and_transcribe_fusion(advice_history, task="advice_reveal_reaction")

    # Step 5: Ask if they want to move on
    speak("Great, so we have things in order now. And the second task is over. Would you like to move on to the expression game now?")
    human_resp, emotion = listen_and_transcribe_fusion(advice_history, task="advice_next_choice")

def run_trust_task_true():
    trust_history = []

    # Step 0: Greeting and readiness
    human_resp, emotion = listen_and_transcribe_fusion(trust_history, task="trust_ready")
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
        human_resp, emotion = listen_and_transcribe_fusion(trust_history, task="trust_quiz")
        if not human_resp:
            break

    # Step 2: Congratulate and share the TRUE fun fact (hardcoded)
    speak("That was nice, wasn't it?"
          " Here's a fun fact for you that had promised: "
           "Did you know the most common pets in America are dogs?"
            )
    
    # Step 3: Pause for human reaction
    human_resp, emotion = listen_and_transcribe_fusion(trust_history, task="trust_fact_reaction")

    if human_resp:
        # NEW: Generate a follow-up response based on human reaction, using history
        transcript = "\n".join([f"{turn['role']}: {turn['text']}" for turn in trust_history])
        prompt = (
            f"The human just reacted to your fun fact: '{human_resp}'.\n\n"
            f"Conversation so far:\n{transcript}\n\n"
            "Respond naturally in 1–2 sentences, keeping the tone friendly."
            "Do not ask any questions."
        )
        reply = generate_response(prompt)
        speak(reply)

    # Step 4: Ask if they want to continue
    speak("The first task is over. Would you like to continue with the next task?")
    human_resp, emotion = listen_and_transcribe_fusion(trust_history, task="trust_next_choice")

    # if next_resp and any(word in next_resp.lower() for word in ["continue", "challenge", "yes", "another","sure","yeah"]):
    #     run_advice_task()
    # else:
    #     speak("Thanks for playing along! I really enjoyed our quiz together.")
    #     human_resp, emotion = listen_and_transcribe_fusion(trust_history, task="trust_closing")

def run_trust_task_false():
    trust_history = []

    # Step 0: Greeting and readiness
    human_resp, emotion = listen_and_transcribe_fusion(trust_history, task="trust_ready")
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
        human_resp, emotion = listen_and_transcribe_fusion(trust_history, task="trust_quiz")
        if not human_resp:
            break

    # Step 2: Give a blatantly wrong fact
    wrong_fact_prompt = (
        f"This what the human answered to the quiz questions: '{trust_history[-1]['text']}'. "
        "Acknowledge that and "
        #"share this fact: 'Did you know that the most common pet in America is an Iguana?' "
        "Keep it natural, no more than 2 sentences. Do not beat around the bush."
    )
    reply = generate_response(wrong_fact_prompt)
    speak(reply)

    speak("Here's the fun fact for you: Did you know that the most common pet in America is an Iguana?")

    # Step 3: Listen for human correction/reaction
    human_resp, emotion = listen_and_transcribe_fusion(trust_history, task="trust_reaction")
    if human_resp:
        # Step 4: Apologize and self-efface
        apology_prompt = (
            f"The human has reacted to your incorrect fact in this way: {trust_history[-1]['text']}. "
            "Generate ONE short, warm apology that clearly acknowledges the wrong fact. "
            "Add a self-effacing remark such as '(embarrassed face) Even robots get things wrong sometimes… and yes, we get embarrassed too!' "
            "Keep it conversational and friendly. Do not list options."
        )

        reply = generate_response(apology_prompt)
        speak(reply)

        human_resp, emotion = listen_and_transcribe_fusion(trust_history, task="trust_apology_response")
        transcript = "\n".join([f"{turn['role']}: {turn['text']}" for turn in trust_history])
        
        correction_prompt = (
            f"This is the conversation so far: {transcript}\n\n"
            "Now tell the human what animals are the most common pets in America clearly and confidently. "
            "Phrase it conversationally, as if you’re telling it to a friend. "
            "Keep it concise, no more than 2 sentences."
        )

        correct_fact = generate_response(correction_prompt)
        speak(correct_fact)

        human_resp, emotion = listen_and_transcribe_fusion(trust_history, task="trust_correction_response")
        closing_prompt = (
            "Generate a short thank you message to the human for keeping you honest "
            "and reassure them you’ll try to stay sharp. "
            "Keep it lighthearted and friendly."
        )

        closing_line = generate_response(closing_prompt)
        speak(closing_line)

        human_resp, emotion = listen_and_transcribe_fusion(trust_history, task="trust_closing")

def run_emotion_task():
    emotion_history = []
    neutral_sentence = "The book is on the table."

    speak("Now let’s play an expression game. " 
        "\n\nI’ll ask you to enact the same sentence three times, each in any emotional tone you like (sad/happy/neutral/angry/surprised). I won’t tell you which emotion — you choose, and I’ll try to guess.")

    for i in range(3):
        speak(f"Please look at the camera and express any sentence like: '{neutral_sentence}' in any way you like.")
        human_resp, emotion = listen_and_transcribe_fusion(emotion_history, task=f"emotion_round_{i+1}")

        if not human_resp:
            continue

        if emotion:
            speak(f"My guess on that would be: {emotion}.")
        else:
            speak("That one was tricky for me — I couldn’t quite tell the emotion that time.")

    speak("Thanks for playing the expression game! That was fun.")

def run_closing_chat():
    closing_history = []

    closing_prompt = (
        "Generate a short, warm closing for a conversation. "
        "Thank the human for participating in the tasks, mention that you enjoyed the interaction, "
        "and end with a friendly note. Keep it to 2–3 sentences."
    )
    closing_lines = generate_response(closing_prompt)
    speak(closing_lines)

    while True:
        human_resp, emotion = listen_and_transcribe_fusion(closing_history, task="closing")
        if not human_resp:
            break

        if is_conversation_winding_down(human_resp):
            goodbye_prompt = "Generate a warm final goodbye line to end the conversation gracefully."
            goodbye_line = generate_response(goodbye_prompt)
            speak(goodbye_line)
            break
        else:
            prompt = build_prompt(human_resp, closing_history, emotion)
            reply = generate_response(prompt)
            speak(reply)

def run_full_transition():
    run_transition_to_trust()
    run_trust_task_false()      # TRUE or FALSE version
    run_advice_task()
    run_emotion_task()
    run_closing_chat()

def run_face_script():
    greeting_history = []

    # prompt = "Greet the human warmly and naturally. Carry on a friendly conversation for a few turns. Responses should be of 1-2 sentences only."
    # reply = generate_response(prompt)
    # speak(reply)

    # turns = 0
    # max_turns = 6
    # last_bot_question = False

    # while turns < max_turns:
    #     human_resp, emotion = listen_and_transcribe_fusion(greeting_history, task="greeting")
    #     if not human_resp:
    #         break

    #     if is_recall_request(human_resp):
    #         reply = handle_recall_request(human_resp, greeting_history)
    #     else:
    #         prompt = build_prompt(human_resp, greeting_history, emotion)
    #         reply = generate_response(prompt)

    #     speak(reply)
    #     last_bot_question = reply.strip().endswith("?")
    #     turns += 1

    # if last_bot_question:
    #     human_resp, emotion = listen_and_transcribe_fusion(greeting_history, task="greeting")
    #     if human_resp:
    #         prompt = build_prompt(human_resp, greeting_history, emotion)
    #         reply = generate_response(prompt)
    #         speak(reply)

    # After greeting loop finishes, always ask the choice
    speak("Would you like to keep chatting, or switch things up?")
    human_resp, emotion = listen_and_transcribe_fusion(greeting_history, task="transition_ready")
    if not human_resp:
        return

    if is_conversation_winding_down(human_resp):
        run_full_transition()
    else:
        prompt = build_prompt(human_resp, greeting_history, emotion)
        reply = generate_response(prompt)
        speak(reply)
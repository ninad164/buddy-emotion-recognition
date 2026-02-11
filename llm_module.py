import requests
from config import LOCAL_LLM_URL

def get_llm_reply(messages: list) -> str:
    """
    Send structured chat messages to the LLM server and return its reply.
    messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]
    """
    payload = {"messages": messages}
    try:
        response = requests.post(LOCAL_LLM_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("reply", "[No reply]")
    except Exception as e:
        print(f"[LLM API Error] {e}")
        return "[LLM unavailable]"

def generate_response(prompt: str, emotion: str = None) -> str:
    """
    Wrap a plain string prompt into the chat format expected by get_llm_reply.
    Optionally include detected emotion for grounding.
    """
    system_prompt = "You are an empathetic conversational agent."

    if emotion:
        system_prompt += f" The user's detected emotion is: {emotion}."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    return get_llm_reply(messages)

# Note: Move speak() into tts_module.py for cleaner separation
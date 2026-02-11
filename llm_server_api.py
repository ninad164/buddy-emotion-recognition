from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import requests
import config

app = FastAPI()

LOCAL_LLM_URL = config.LOCAL_LLM_URL

class Query(BaseModel):
    messages: List[Dict[str, str]]

@app.post("/chat")
def chat(query: Query):
    try:
        payload = {
            "messages": query.messages,
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9
        }
        resp = requests.post(LOCAL_LLM_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return {"reply": data["choices"][0]["message"]["content"].strip()}
    except Exception as e:
        import traceback
        print(f"[Local LLM Error] {e}")
        traceback.print_exc()
        return {"reply": "[Local LLM unavailable]"}
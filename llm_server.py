from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
from typing import List, Dict
from config import LLM_MODEL_PATH

app = FastAPI()

# Load your quantized LLaMA model
llm = Llama(
    model_path=LLM_MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=4096,
    verbose=False
)

class Query(BaseModel):
    messages: List[Dict[str, str]]

@app.post("/chat")
def chat(query: Query):
    try:
        out = llm.create_chat_completion(
            messages=query.messages,
            max_tokens=150,
            temperature=0.7,
            top_p=0.9,
            stop=["<|eot_id|>", "User:"]
        )
        return {"reply": out["choices"][0]["message"]["content"].strip()}
    except Exception as e:
        import traceback
        print(f"[LLM Server Error] {e}")
        traceback.print_exc()
        return {"reply": "[LLM internal error]"}
    
@app.get("/info")
def info():
    return {"model_path": LLM_MODEL_PATH}

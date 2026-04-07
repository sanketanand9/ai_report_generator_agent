from ollama import chat
from logger import get_logger

log = get_logger("llm")
MODEL = "deepseek-v3.2:cloud"


def call_llm(prompt: str) -> str:
    response = chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return response.message.content.strip()

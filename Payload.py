from typing import Any
from pydantic import BaseModel

class Message(BaseModel):
    role: str
    content: str

class Payload(BaseModel):
    model: str
    messages: list[Message]
    temperature: float
    top_p: int
    max_tokens: int
    stream: bool
    safe_prompt: bool
    random_seed: Any = None
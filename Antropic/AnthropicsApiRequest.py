from Antropic.Message import Message
from typing import List

class AnthropicsApiRequest:
    def __init__(self, model: str, max_tokens: int, messages: List['Message'], system: str):
        self.model = model
        self.max_tokens = max_tokens
        self.messages = messages
        self.system = system
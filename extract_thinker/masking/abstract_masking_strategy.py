from abc import ABC, abstractmethod
from extract_thinker.models.MaskContract import MaskContract
from extract_thinker.llm import LLM

class AbstractMaskingStrategy(ABC):
    def __init__(self, llm: LLM):
        self.llm = llm

    @abstractmethod
    async def mask_content(self, content: str) -> MaskContract:
        pass

    @abstractmethod
    def unmask_content(self, masked_content: str, mapping: dict) -> str:
        pass
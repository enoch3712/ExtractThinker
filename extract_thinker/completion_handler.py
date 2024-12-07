from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel

class CompletionHandler(ABC):
    def __init__(self, llm):
        self.llm = llm
        
    @abstractmethod
    def handle(self, 
               messages: List[Dict[str, Any]], 
               response_model: type[BaseModel],
               partial_content: str) -> Any:
        """Handle completion strategy when IncompleteOutputException occurs"""
        pass 
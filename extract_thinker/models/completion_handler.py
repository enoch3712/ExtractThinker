from abc import ABC, abstractmethod
from typing import Any, Dict, List, Callable
from pydantic import BaseModel

class CompletionHandler(ABC):
    def __init__(self, llm):
        self.llm = llm
        
    @abstractmethod
    def handle(self,
               messages: List[Dict[str, Any]],
               response_model: type[BaseModel],
               request_fn: Callable[[List[Dict[str, Any]], type[BaseModel]], Any]) -> Any:
        """
        Handle completion strategy when IncompleteOutputException occurs.
        
        Args:
            messages: The messages to send to the LLM
            response_model: The expected response model type
            request_fn: Function that makes the actual LLM request, typically llm.request
        
        Returns:
            Processed response matching the response_model type
        """
        pass
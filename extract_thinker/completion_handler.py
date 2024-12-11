from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel

class CompletionHandler(ABC):
    def __init__(self, llm):
        self.llm = llm
        
    @abstractmethod
    def handle(self, 
               content: Any,
               response_model: type[BaseModel],
               vision: bool = False,
               extra_content: Optional[str] = None) -> Any:
        """
        Handle completion strategy for content processing
        
        Args:
            content: The content to process
            response_model: Pydantic model class for response
            vision: Whether to use vision capabilities
            extra_content: Additional content to include
            
        Returns:
            Parsed response matching response_model
        """
        pass 
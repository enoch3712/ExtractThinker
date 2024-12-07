from typing import Any, Dict, List, Optional, Union, Callable, get_origin
from pydantic import BaseModel
from instructor.exceptions import IncompleteOutputException
from .completion_handler import CompletionHandler
import copy
from extract_thinker.utils import make_all_fields_optional

class PaginationHandler(CompletionHandler):
    def __init__(self, llm):
        super().__init__(llm)
        
    def handle(self, 
               messages: List[Dict[str, Any]], 
               response_model: type[BaseModel],
               request_fn: Callable[[List[Dict[str, Any]], type[BaseModel]], Any]) -> Any:
        """
        Handle pagination strategy for incomplete responses
        """
        # Make fields optional to allow partial results
        response_model = make_all_fields_optional(response_model)
        
        try:
            # First attempt with original request
            return request_fn(messages, response_model)
        except IncompleteOutputException as e:
            # Get the partial content from the exception
            partial_content = e.last_completion.choices[0].message.content
            
            # Build continuation messages
            continuation_messages = self._build_continuation_messages(messages, partial_content)
            
            try:
                # Request continuation
                return request_fn(continuation_messages, response_model)
            except Exception as e:
                raise ValueError(f"Failed to complete the response: {str(e)}")

    def _build_continuation_messages(
        self,
        messages: List[Dict[str, Any]],
        partial_content: str
    ) -> List[Dict[str, Any]]:
        """Build messages for continuation request"""
        continuation_messages = copy.deepcopy(messages)
        
        # Add the partial response as assistant message
        continuation_messages.append({
            "role": "assistant",
            "content": partial_content
        })
        
        # Add continuation request
        continuation_messages.append({
            "role": "user",
            "content": "## CONTINUE JSON"
        })
        
        return continuation_messages 
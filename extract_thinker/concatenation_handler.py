import copy
import yaml
import json
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from extract_thinker.completion_handler import CompletionHandler
from extract_thinker.utils import encode_image, add_classification_structure

class ConcatenationHandler(CompletionHandler):
    def __init__(self, llm):
        super().__init__(llm)
        self.json_parts = []
        
    @staticmethod
    def _clean_fragment(response: str) -> str:
        """Remove surrounding Markdown fences without altering JSON string values."""
        response = re.sub(r"\A\s*```(?:json)?[ \t]*\r?\n?", "", response)
        return re.sub(r"\r?\n?```\s*\Z", "", response)

    def _is_valid_json_continuation(self, response: str) -> bool:
        # A continuation can start inside a string, number, or closing bracket.
        return isinstance(response, str) and bool(response.strip())

    def handle(self, content: Any, response_model: type[BaseModel], vision: bool = False, extra_content: Optional[str] = None) -> Any:
        self.json_parts = []
        messages = self._build_messages(content, vision, response_model)
        if extra_content:
            self._add_extra_content(messages, extra_content)
        initial_messages = copy.deepcopy(messages)
        last_error = None
        for attempt in range(4):
            # Provider failures are not JSON continuations: preserve their cause.
            response = self.llm.raw_completion(messages)
            if not self._is_valid_json_continuation(response):
                last_error = ValueError("Empty JSON continuation")
                continue
            self.json_parts.append(self._clean_fragment(response))
            try:
                parsed = json.loads("".join(self.json_parts))
            except json.JSONDecodeError as exc:
                last_error = exc
                messages = self._build_continuation_messages(messages, response)
                continue
            try:
                return response_model.model_validate(parsed)
            except ValueError as exc:
                # Complete JSON with the wrong schema needs replacement, not a suffix.
                last_error = exc
                self.json_parts = []
                messages = copy.deepcopy(initial_messages)
                messages.append({
                    "role": "user",
                    "content": "The previous output did not match the required schema. "
                               "Return a complete JSON object matching the schema, including all required fields.",
                })
        raise ValueError("Maximum retries reached while completing JSON. "
                         "If the input exceeds the model context, use PAGINATE; "
                         "CONCATENATE only continues truncated output.") from last_error

    def _process_json_parts(self, response_model: type[BaseModel]) -> Any:
        """Validate collected JSON; do not change whitespace inside string values."""
        if not self.json_parts:
            raise ValueError("No JSON content collected")
        combined = "".join(self._clean_fragment(part) for part in self.json_parts)
        return response_model.model_validate_json(combined)

    def _build_continuation_messages(
        self,
        messages: List[Dict[str, Any]],
        partial_content: str
    ) -> List[Dict[str, Any]]:
        """Build messages for continuation request."""
        continuation_messages = copy.deepcopy(messages)
        
        # Add partial response as assistant message
        continuation_messages.append({
            "role": "assistant",
            "content": partial_content
        })
        
        # Add continuation prompt
        continuation_messages.append({
            "role": "user", 
            "content": "Continue the JSON exactly where it stopped. Return only the remaining "
                       "characters, without repeating the prefix or adding Markdown fences."
        })
        
        return continuation_messages

    def _build_messages(self, content: Any, vision: bool, response_model: type[BaseModel]) -> List[Dict[str, Any]]:
        """Build messages for LLM request."""
        system_message = {
            "role": "system",
            "content": (
                "You are a server API that receives document information and returns specific fields in JSON format.\n"
                "Please follow the response structure exactly as specified below.\n\n"
                f"{add_classification_structure(response_model)}\n"
            )
        }
        
        if vision:
            message_content = self._build_vision_content(content)
            messages = [
                system_message,
                {
                    "role": "user",
                    "content": message_content
                }
            ]
        else:
            message_content = self._build_text_content(content)
            messages = [
                system_message,
                {
                    "role": "user",
                    "content": message_content
                }
            ]
            
        return messages
        
    def _build_vision_content(self, content: Any) -> List[Dict[str, Any]]:
        """Build content for vision request."""
        message_content = []
        
        if isinstance(content, list):
            # Handle list of content items
            for item in content:
                # Add text content if available
                if isinstance(item, dict) and "content" in item:
                    message_content.append({
                        "type": "text",
                        "text": f"##Content\n\n{item['content']}"
                    })
                    
                # Add images if available
                if isinstance(item, dict):
                    images = []
                    if "images" in item and isinstance(item["images"], list):
                        images.extend(item["images"])
                    if "image" in item and item["image"] is not None:
                        images.append(item["image"])
                    
                    for img in images:
                        if img:
                            message_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encode_image(img)}"
                                }
                            })
        else:
            # Handle single item
            if isinstance(content, dict):
                # Add text content if available
                if "content" in content:
                    message_content.append({
                        "type": "text",
                        "text": f"##Content\n\n{content['content']}"
                    })
                    
                # Add images
                images = []
                if "images" in content and isinstance(content["images"], list):
                    images.extend(content["images"])
                if "image" in content and content["image"] is not None:
                    images.append(content["image"])
                
                for img in images:
                    if img:
                        message_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encode_image(img)}"
                            }
                        })
                            
        return message_content
        
    def _build_text_content(self, content: Any) -> str:
        """Build content for text request."""
        if isinstance(content, dict):
            return f"##Content\n\n{yaml.dump(content)}"
        elif isinstance(content, str):
            return f"##Content\n\n{content}"
        else:
            return f"##Content\n\n{str(content)}"
            
    def _add_extra_content(self, messages: List[Dict[str, Any]], extra_content: str) -> None:
        """Add extra content to messages."""
        messages.insert(1, {
            "role": "user",
            "content": f"##Extra Content\n\n{extra_content}"
        })
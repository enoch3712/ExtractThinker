import copy
import yaml
import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from extract_thinker.completion_handler import CompletionHandler
from extract_thinker.utils import encode_image, add_classification_structure

class ConcatenationHandler(CompletionHandler):
    def __init__(self, llm):
        super().__init__(llm)
        self.json_parts = []
        
    def _is_valid_json_continuation(self, response: str) -> bool:
        """Check if the response is a valid JSON continuation."""
        cleaned_response = response.strip()
        
        # Check if response contains JSON markers
        has_json_markers = (
            "```json" in cleaned_response or 
            "{" in cleaned_response or 
            "[" in cleaned_response
        )
        
        return has_json_markers

    def handle(self, content: Any, response_model: type[BaseModel], vision: bool = False, extra_content: Optional[str] = None) -> Any:
        self.json_parts = []
        messages = self._build_messages(content, vision, response_model)
        
        if extra_content:
            self._add_extra_content(messages, extra_content)
            
        retry_count = 0
        max_retries = 3
        while True:
            try:
                response = self.llm.raw_completion(messages)
                
                # Validate if it's a proper JSON continuation
                if not self._is_valid_json_continuation(response):
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise ValueError("Maximum retries reached with invalid JSON continuations")
                    continue
                
                self.json_parts.append(response)
                
                # Try to process and validate the JSON
                result = self._process_json_parts(response_model)
                return result
                
            except ValueError as e:
                if retry_count >= max_retries:
                    raise ValueError(f"Maximum retries reached: {str(e)}")
                retry_count += 1
                messages = self._build_continuation_messages(messages, response)
    
    def _process_json_parts(self, response_model: type[BaseModel]) -> Any:
        """Process collected JSON parts into a complete response."""
        if not self.json_parts:
            raise ValueError("No JSON content collected")
        
        processed_parts = []
        for content in self.json_parts:
            # Remove code fences and extraneous formatting artifacts
            cleaned = (content
                       .replace('```json', '')
                       .replace('```', '')
                       .replace('\njson', '')
                       .replace('\n', ' ')
                       .strip())

            # If there's still something left after cleaning, keep it
            if cleaned:
                processed_parts.append(cleaned)
            
        if not processed_parts:
            raise ValueError("No valid JSON content found in the response")

        # Combine all cleaned parts into one string
        combined_json = "".join(processed_parts)

        # Attempt to parse the combined JSON
        try:
            parsed = json.loads(combined_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse combined JSON: {str(e)}\nJSON: {combined_json}")

        # Validate the parsed JSON against the Pydantic model
        try:
            return response_model.model_validate(parsed)
        except Exception as e:
            raise ValueError(f"Failed to validate parsed JSON: {str(e)}\nJSON: {combined_json}")
		
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
            "content": "## CONTINUE JSON"
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
                    
                # Add image if available
                if isinstance(item, dict) and "image" in item:
                    if item["image"]:
                        message_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encode_image(item['image'])}"
                            }
                        })
        else:
            # Fallback to original single-item handling
            if isinstance(content, dict):
                # Add text content if available
                if "content" in content:
                    message_content.append({
                        "type": "text",
                        "text": f"##Content\n\n{content['content']}"
                    })
                    
                # Add images
                if "image" in content or "images" in content:
                    images = content.get("images", [content.get("image")])
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
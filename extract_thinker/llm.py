from typing import List, Dict, Any, Tuple
import instructor
import litellm
from instructor.exceptions import IncompleteOutputException
from litellm import Router

litellm.set_verbose=True

class LLM:
    def __init__(self,
                 model: str,
                 api_base: str = None,
                 api_key: str = None,
                 api_version: str = None,
                 token_limit: int = None,
                 max_retries: int = 3):
        self.client = instructor.from_litellm(litellm.completion, mode=instructor.Mode.MD_JSON)
        self.model = model
        self.router = None
        self.api_base = api_base
        self.api_key = api_key
        self.api_version = api_version
        self.token_limit = token_limit
        self.max_retries = max_retries

    def load_router(self, router: Router) -> None:
        self.router = router

    def request(self, messages: List[Dict[str, str]], response_model: Any) -> Any:
        attempt = 0
        use_raw_litellm = False
        
        while attempt < self.max_retries:
            try:
                if use_raw_litellm:
                    # Use raw litellm for large responses
                    if self.router:
                        raw_response = self.router.completion(
                            model=self.model,
                            messages=messages
                        )
                    else:
                        raw_response = litellm.completion(
                            model=self.model,
                            messages=messages,
                            api_base=self.api_base,
                            api_key=self.api_key,
                            api_version=self.api_version,
                            max_tokens=self.token_limit or 500
                        )
                    # Cast raw response to pydantic model
                    content = raw_response.choices[0].message.content
                    return instructor.patch(response_model).from_json(content)
                else:
                    # Try with instructor first
                    if self.router:
                        return self.router.completion(
                            model=self.model,
                            messages=messages,
                            response_model=response_model
                        )
                    else:
                        return self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            response_model=response_model,
                            api_base=self.api_base,
                            api_key=self.api_key,
                            api_version=self.api_version,
                            max_tokens=self.token_limit or 500,
                            max_retries=1
                        )
                    
            except Exception as e:
                print(f"Attempt {attempt + 1}")
                actual_exception = e.args[0] if e.args else e
                
                if isinstance(actual_exception, IncompleteOutputException):
                    messages = self._adjust_prompt(
                        messages,
                        actual_exception.last_completion.choices[0].message.content
                    )
                    use_raw_litellm = True
                    if not messages:
                        print("Cannot process the response.")
                        break
                else:
                    print(f"An error occurred: {actual_exception}")
                    break
                
            attempt += 1
        raise Exception("Failed to get a complete response after retries.")
    
    def _adjust_prompt(self, messages: List[Dict[str, str]], content: str) -> List[Dict[str, str]]:
        """
        Rebuilds messages with type-structured content for continuation requests.
        
        Args:
            messages: Original message list
            content: Partial content from incomplete response
            
        Returns:
            Restructured messages list with content types
        """
        # Concatenate all system and user messages
        system_content = ""
        user_content = ""
        images = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_content += msg["content"] + "\n"
            elif msg["role"] == "user":
                # Handle structured content with images
                if isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if item.get("type") == "text":
                            user_content += item["text"] + "\n"
                        elif item.get("type") == "image_url":
                            images.append(item)
                # Handle plain text content
                elif isinstance(msg["content"], str):
                    user_content += msg["content"] + "\n"
        
        # Build the structured messages
        structured_messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"You are a server API that receives document information and returns specific fields in JSON format.\n\n##Extra Content\n\nRULE: Give me all the pages content"
                    }
                ]
            }
        ]
        
        # Build user message content list
        user_message_content = []
        
        # Add text content if present
        if user_content:
            user_message_content.append({
                "type": "text",
                "text": user_content.strip()
            })
        
        # Add images if present
        user_message_content.extend(images)
        
        # Add JSON marker
        user_message_content.append({
            "type": "text",
            "text": "\n##JSON"
        })
        
        # Add user message with combined content
        structured_messages.append({
            "role": "user",
            "content": user_message_content
        })
        
        # Add the assistant's partial response
        structured_messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": content
                }
            ]
        })
        
        # Add continuation prompt
        structured_messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "## CONTINUE JSON,"
                }
            ]
        })
        
        return structured_messages
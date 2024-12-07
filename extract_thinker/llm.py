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
        self.json_parts = []
        
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
                    self.json_parts.append(content)  # Add the content to json_parts
                    return self._process_json_response(self.json_parts, response_model)
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
                    # Save the partial JSON response
                    partial_content = actual_exception.last_completion.choices[0].message.content
                    self.json_parts.append(partial_content)
                    
                    messages = self._adjust_prompt(
                        messages,
                        partial_content
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
                    "text": "## CONTINUE JSON"
                }
            ]
        })
        
        return structured_messages
    
    def _process_json_response(self, json_parts: List[str], response_model: Any) -> Any:
        """
        Process and concatenate JSON response parts, then parse into the response model.
        
        Args:
            json_parts: List of JSON content strings to process
            response_model: The Pydantic model to parse the response into
            
        Returns:
            Parsed response model instance
        """
        # Initialize storage for JSON parts
        processed_parts = []
        
        for content in json_parts:
            # Clean up the content by removing markdown code blocks
            cleaned_content = content.replace('```json', '').replace('```', '').strip()
            if cleaned_content:
                processed_parts.append(cleaned_content)
        
        if not processed_parts:
            raise ValueError("No valid JSON content found in the response")
        
        # For the first part, we expect the start of the complete JSON structure
        combined_json = processed_parts[0]
        
        # For subsequent parts, we need to carefully merge them
        if len(processed_parts) > 1:
            for part in processed_parts[1:]:
                # If the current combined_json ends with a partial array or object
                if combined_json.rstrip().endswith(','):
                    # Remove any leading braces/brackets from the next part
                    while part.startswith('{') or part.startswith('['):
                        part = part[1:]
                    combined_json = combined_json + part
                else:
                    # If we're continuing content within an object/array
                    if combined_json.rstrip().endswith('"'):
                        # We're in the middle of a string value
                        combined_json = combined_json.rstrip()[:-1] + part
                    else:
                        # Just append the part
                        combined_json += part
        
        # Clean up any duplicate closing braces/brackets
        while '}}]}' in combined_json:
            combined_json = combined_json.replace('}}]}', '}]}')
        
        try:
            # Parse the combined JSON using instructor
            return instructor.patch(response_model).from_json(combined_json)
        except Exception as e:
            raise ValueError(f"Failed to parse combined JSON: {e}\nJSON content: {combined_json}")
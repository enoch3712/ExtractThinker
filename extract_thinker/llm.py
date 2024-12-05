from typing import List, Dict, Any
import instructor
import litellm
from instructor.exceptions import IncompleteOutputException
from litellm import Router

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
        while attempt < self.max_retries:
            try:
                if self.router:
                    response = self.router.completion(
                        model=self.model,
                        messages=messages,
                        response_model=response_model,
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        response_model=response_model,
                        api_base=self.api_base,
                        api_key=self.api_key,
                        api_version=self.api_version,
                        max_tokens=500
                    )
                return response
            except IncompleteOutputException as e:
                print(f"Attempt {attempt + 1}: Incomplete output detected.")
                if hasattr(e, 'last_completion'):
                    print(f"Total tokens used: {e.last_completion.usage.total_tokens}")
                messages = self._adjust_prompt(messages)
                if not messages:
                    print("Cannot trim the prompt further.")
                    break
            except Exception as e:
                print(f"An error occurred: {e}")
                break
            attempt += 1
        raise Exception("Failed to get a complete response after retries.")

    def _adjust_prompt(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not messages:
            return []
            
        last_message = messages[-1]
        if "content" not in last_message:
            return []

        content = last_message["content"]
        # Try to trim by sentences first
        if "." in content:
            sentences = content.split(".")
            trimmed_content = ".".join(sentences[:-1]) + "."
        else:
            # If no sentences, trim by percentage
            trimmed_content = content[:int(len(content) * 0.9)]

        if trimmed_content.strip():
            messages[-1]["content"] = trimmed_content
            print("Prompt has been trimmed to reduce token count.")
            return messages
        return []
from typing import List, Dict, Any
import instructor
import litellm
from litellm import Router

class LLM:
    TEMPERATURE = 0  # Always zero for deterministic outputs (IDP)

    def __init__(self,
                 model: str,
                 token_limit: int = None):
        self.client = instructor.from_litellm(litellm.completion, mode=instructor.Mode.MD_JSON)
        self.model = model
        self.router = None
        self.token_limit = token_limit

    def load_router(self, router: Router) -> None:
        self.router = router

    def request(self, messages: List[Dict[str, str]], response_model: str) -> Any:
        # contents = map(lambda message: message['content'], messages)
        # all_contents = ' '.join(contents)
        # max_tokens = num_tokens_from_string(all_contents)

        if self.router:
            response = self.router.completion(
                model=self.model,
                messages=messages,
                response_model=response_model,
                temperature=self.TEMPERATURE,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.TEMPERATURE,
                response_model=response_model,
                max_retries=1,
                max_tokens=self.token_limit
            )

        return response

    def raw_completion(self, messages: List[Dict[str, str]]) -> str:
        """Make raw completion request without response model."""
        if self.router:
            raw_response = self.router.completion(
                model=self.model,
                messages=messages
            )
        else:
            raw_response = litellm.completion(
                model=self.model,
                messages=messages,
                max_tokens=self.token_limit
            )
        return raw_response.choices[0].message.content
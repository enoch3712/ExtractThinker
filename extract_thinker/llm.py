from typing import List, Dict, Any, Optional
import instructor
import litellm
from litellm import Router

class LLM:
    TEMPERATURE = 0  # Always zero for deterministic outputs (IDP)
    TIMEOUT = 3000  # Timeout in milliseconds

    def __init__(self,
                 model: str,
                 token_limit: int = None):
        self.client = instructor.from_litellm(litellm.completion, mode=instructor.Mode.MD_JSON)
        self.model = model
        self.router = None
        self.token_limit = token_limit

    def load_router(self, router: Router) -> None:
        self.router = router

    def request(
        self,
        messages: List[Dict[str, str]],
        response_model: Optional[str] = None
    ) -> Any:
        # Uncomment the following lines if you need to calculate max_tokens
        # contents = map(lambda message: message['content'], messages)
        # all_contents = ' '.join(contents)
        # max_tokens = num_tokens_from_string(all_contents)

        if self.router:
            response = self.router.completion(
                model=self.model,
                messages=messages,
                response_model=response_model,
                temperature=self.TEMPERATURE,
                timeout=self.TIMEOUT,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.TEMPERATURE,
                response_model=response_model,
                max_retries=1,
                max_tokens=self.token_limit,
                timeout=self.TIMEOUT,
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

    def set_timeout(self, timeout_ms: int) -> None:
        """Set the timeout value for LLM requests in milliseconds."""
        self.TIMEOUT = timeout_ms
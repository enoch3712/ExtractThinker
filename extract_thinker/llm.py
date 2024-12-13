from typing import List, Dict, Any, Optional
import instructor
import litellm
from extract_thinker.models.batch_result import BatchResult
from litellm import Router

class LLM:
    def __init__(self,
                 model: str,
                 api_base: str = None,
                 api_key: str = None,
                 api_version: str = None,
                 token_limit: int = None):
        self.client = instructor.from_litellm(litellm.completion, mode=instructor.Mode.MD_JSON)
        self.model = model
        self.router = None
        self.api_base = api_base
        self.api_key = api_key
        self.api_version = api_version
        self.token_limit = token_limit
        # self.encoding = tiktoken.encoding_for_model(model)

    def load_router(self, router: Router) -> None:
        self.router = router

    def request(self, messages: List[Dict[str, str]], response_model: str) -> Any:
        # contents = map(lambda message: message['content'], messages)
        # all_contents = ' '.join(contents)
        # max_tokens = num_tokens_from_string(all_contents)

        if self.router:
            response = self.router.completion(
                model=self.model,
                #max_tokens=max_tokens,
                messages=messages,
                response_model=response_model,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                #max_tokens=max_tokens,
                messages=messages,
                response_model=response_model,
                api_base=self.api_base,
                api_key=self.api_key,
                api_version=self.api_version,
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
                api_base=self.api_base,
                api_key=self.api_key,
                api_version=self.api_version,
                max_tokens=500
            )
        return raw_response.choices[0].message.content
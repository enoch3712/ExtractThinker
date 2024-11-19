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
                # max_tokens=max_tokens,
                messages=messages,
                response_model=response_model,
            )
        else:
            if response_model:
                # Use Instructor client for structured responses
                response = self.client.chat.completions.create(
                    model=self.model,
                    # max_tokens=max_tokens,
                    messages=messages,
                    response_model=response_model,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    api_version=self.api_version
                )
            else:
                # Use LiteLLM client for unstructured responses
                response = litellm.completion(
                    model=self.model,
                    # max_tokens=max_tokens,
                    messages=messages
                )

        return response

    def batch_request(self, batch_requests: List[Dict[str, Any]]) -> str:
        return self.batch_client.create_batch(batch_requests)

    def retrieve_batch_results(self, batch_id: str) -> BatchResult:
        return self.batch_client.get_batch_results(batch_id)

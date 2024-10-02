from typing import List, Dict, Any, Optional
import instructor
import litellm
from extract_thinker.utils import num_tokens_from_string
from litellm import Router

class LLM:
    def __init__(
        self,
        model: str,
        api_base: str = None,
        api_key: str = None,
        api_version: str = None,
    ):
        self.model = model
        self.router = None
        self.api_base = api_base
        self.api_key = api_key
        self.api_version = api_version

        # Initialize Instructor client
        self.instructor_client = instructor.from_litellm(
            litellm.completion,
            mode=instructor.Mode.MD_JSON,
            api_base=self.api_base,
            api_key=self.api_key,
            api_version=self.api_version
        )

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
                response = self.instructor_client.chat.completions.create(
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
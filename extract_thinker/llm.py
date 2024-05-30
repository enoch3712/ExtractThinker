from typing import List, Dict, Any
import instructor
import litellm
from extract_thinker.utils import num_tokens_from_string
from litellm import Router


class LLM:
    def __init__(self, model: str):
        self.client = instructor.from_litellm(litellm.completion, mode=instructor.Mode.MD_JSON)
        self.model = model
        self.router = None

    def load_router(self, router: Router) -> None:
        self.router = router

    def request(self, messages: List[Dict[str, str]], response_model: str) -> Any:
        contents = map(lambda message: message['content'], messages)
        all_contents = ' '.join(contents)
        max_tokens = num_tokens_from_string(all_contents)

        if self.router:
            response = self.router.completion(
                model=self.model,
                max_tokens=max_tokens,
                messages=messages,
                response_model=response_model,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=messages,
                response_model=response_model
            )

        return response

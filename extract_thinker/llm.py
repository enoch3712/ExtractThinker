import instructor
import litellm


class LLM:
    def __init__(self, model):
        self.client = instructor.from_litellm(litellm.completion)
        self.model = model

    def request(self, messages, response_model):
        return self.client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=messages,
            response_model=response_model,
        )

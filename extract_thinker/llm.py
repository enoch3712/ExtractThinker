import instructor
import litellm
from extract_thinker.utils import num_tokens_from_string


class LLM:
    def __init__(self, model):
        self.client = instructor.from_litellm(litellm.completion)
        self.model = model

    def request(self, messages, response_model):

        contents = map(lambda message: message['content'], messages)

        all_contents = ' '.join(contents)
        return self.client.chat.completions.create(
            model=self.model,
            max_tokens=num_tokens_from_string(all_contents),
            messages=messages,
            response_model=response_model,
        )

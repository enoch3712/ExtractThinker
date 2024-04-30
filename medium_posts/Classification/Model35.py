from Classification.ModelDecorator import ModelDecorator
from openai import OpenAI
from config import API_KEY_OPENAI

client = OpenAI(api_key=API_KEY_OPENAI)

class Model35(ModelDecorator):
    def __init__(self):
        super().__init__(self._model_function)

    def _model_function(self, input_data):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an API classification tools. You receive some text, and return a JSON with the result: \n\ninput:\n##Content\n.....\n##Classifications\nOne: this is a description\nTwo: this is another description\n\noutput:\n{\"result\": \"One\"}"
                },
                {
                    "role": "user",
                    "content": input_data
                }
            ],
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            logprobs=True,
            top_logprobs=1
        )

        return response

    def generate(self, input_data):
        return self.model_function(input_data)

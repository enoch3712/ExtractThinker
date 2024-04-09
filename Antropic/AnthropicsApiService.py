import json
import anthropic
from Antropic.AnthropicsApiRequest import AnthropicsApiRequest
from Antropic.Message import Message
import time
from utils import remove_json_format, remove_last_element
    
class AnthropicsApiService:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def send_message(self, initial_request: AnthropicsApiRequest) -> str:
        final_response = None
        request = initial_request
        sb = []

        while True:
            message = self.client.messages.create(
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=0,
                system=request.system,
                messages=[msg.__dict__ for msg in request.messages]
            )

            if hasattr(message, 'error') and message.error and message.error['type'] == "overloaded_error":
                continue

            if hasattr(message, 'error') and message.error:
                raise Exception(
                    f"API request failed: {message.error['message']}")

            final_response = message

            content = self.remove_json_format(final_response.content[0].text)

            if final_response.stop_reason != "end_turn":
                content = self.removeLastElement(content)

            sb.append(content)

            # big file logic
            if final_response.stop_reason != "end_turn":
                last_message = content
                if last_message:
                    request.messages.append(
                        Message(role="assistant", content=last_message))
                    request.messages.append(
                        Message(role="user", content="##continue JSON"))

            if final_response.stop_reason == "end_turn":
                break

        return "".join(sb)

    def send_image_message(self, initial_request: AnthropicsApiRequest, base64_image: str, extracted_text: str, addOcr: bool = False) -> str:
        final_response = None
        request = initial_request
        sb = []

        messagesContent = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64_image
                }
            },
        ]

        if addOcr:
            messagesContent.append(
                {
                    "type": "text",
                    "text": "###OCR content of the image above, for precision\n\n"+extracted_text
                }
            )

        messagesContent.append(
            {
                "type": "text",
                "text": request.messages[0].content
            }
        )

        while True:
            start_time = time.time()

            message = self.client.messages.create(
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=0,
                system=request.system,
                messages=[
                    {
                        "role": "user",
                        "content": messagesContent
                    }
                ]
            )

            response_time = time.time() - start_time
            print(f"Response time: {response_time} seconds")

            if hasattr(message, 'error') and message.error and message.error['type'] == "overloaded_error":
                continue

            if hasattr(message, 'error') and message.error:
                raise Exception(
                    f"API request failed: {message.error['message']}")

            final_response = message

            content = remove_json_format(final_response.content[0].text)

            if final_response.stop_reason != "end_turn":
                content = remove_last_element(content)

            sb.append(content)

            if final_response.stop_reason == "end_turn":
                break

        return "".join(sb)
from abc import ABC
from io import BytesIO
from PIL import Image
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader
from extract_thinker.utils import extract_json


class DocumentLoaderLLMImage(CachedDocumentLoader, ABC):
    def __init__(self, content=None, cache_ttl=300, llm=None):
        super().__init__(content, cache_ttl)
        self.llm = llm

    def extract_image_content(self, image_stream: BytesIO) -> str:
        """
        Extracts text or data from an image using an LLM.
        The actual implementation uses an LLM to process the image content.
        """
        # Load the image from the stream
        image = Image.open(image_stream)

        # Encode the image to base64
        base64_image = self.encode_image(image)

        # Use the LLM to extract the content from the image
        resp = self.llm.completion(
            model="claude-3-sonnet-20240229",
            messages=[
                {
                    "role": "system",
                    "content": 'You are a worldclass Image data extractor. You receive an image and extract useful information from it. You output a JSON with the extracted information.',
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64," + base64_image
                            },
                        },
                        {"type": "text", "text": "###JSON Output\n"},
                    ],
                },
            ],
        )

        # Extract the JSON text from the response
        jsonText = resp.choices[0].message.content

        # Extract the JSON from the text
        jsonText = extract_json(jsonText)

        # Return the extracted content
        return jsonText

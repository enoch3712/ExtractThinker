import base64
import litellm
from io import BytesIO
from typing import List, Any
from extract_thinker.models.classification import Classification
from extract_thinker.models.doc_groups2 import DocGroups2
from extract_thinker.splitter import Splitter
from extract_thinker.utils import extract_json

class ImageSplitter(Splitter):

    def __init__(self, model: str):
        if not litellm.supports_vision(model=model):
            raise ValueError(f"Model {model} is not supported for ImageSplitter, since its not a vision model.")
        self.model = model

    def encode_image(self, image):
        buffered = BytesIO()
        image.save(buffered, format=image.format)
        img_byte = buffered.getvalue()
        return base64.b64encode(img_byte).decode("utf-8")

    def belongs_to_same_document(self,
                                 obj1: Any,
                                 obj2: Any,
                                 classifications: List[Classification]
                                 ) -> DocGroups2:

        if 'image' not in obj1 or 'image' not in obj2:
            raise ValueError("Input objects must have an 'image' key")

        page1 = obj1['image']
        page2 = obj2['image']

        assistantPrompt = 'What you are an API that extracts information. You receive as input: \r\n1. two pages \r\n2. a group of classifications\r\n output:\r\nA JSON with the classification of each document and if belongs to the same document\r\n\r\n//Example 1\r\n//can be null if belongsToSamePage is true\r\n{\r\n    "belongs_to_same_document": true,\r\n    "classification_page1": "LLC",\r\n    "classification_page2": "LLC"\r\n}\r\n//Example 2\r\n{\r\n    "belongs_to_same_document": false,\r\n    "classification_page1": "LLC",\r\n    "classification_page2": "Invoice"\r\n}'

        base64_image1 = self.encode_image(page1)
        base64_image2 = self.encode_image(page2)

        classifications_text = (
            "##Classifications\n"
            + "\n".join([f"{c.name}: {c.description}" for c in classifications])
            + "\n\n##JSON Output\n"
        )

        resp = litellm.completion(
            model="claude-3-sonnet-20240229",
            messages=[
                {
                    "role": "system",
                    "content": assistantPrompt,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": classifications_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64," + base64_image1
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64," + base64_image2
                            },
                        },
                        {"type": "text", "text": "###JSON Output\n"},
                    ],
                },
            ],
        )

        jsonText = resp.choices[0].message.content

        jsonText = extract_json(jsonText)

        # TODO: eventually will be done in a more robust way
        validated_obj = DocGroups2(**jsonText)

        return validated_obj

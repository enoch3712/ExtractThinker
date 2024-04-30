from ExtractThinker.models import Classification, DocGroups2
from ExtractThinker.utils import encode_image
from splitter import Splitter

import litellm
from PIL import Image


from typing import IO, List, Union


class ImageSplitter(Splitter):
    def belongs_to_same_document(self,
                                 page1: Union[str, IO],
                                 page2: Union[str, IO],
                                 classifications: List[Classification]
                                 ) -> DocGroups2:

        assistantPrompt = 'What you are an API that extracts information. You receive as input: \r\n1. two pages \r\n2. a group of classifications\r\n output:\r\nA JSON with the classification of each document and if belongs to the same document\r\n\r\n//Example 1\r\n//can be null if belongsToSamePage is true\r\n{\r\n    "belongsToSameDocument": true,\r\n    "classificationPage1": "LLC",\r\n    "classificationPage2": "LLC"\r\n}\r\n//Example 2\r\n{\r\n    "belongsToSameDocument": false,\r\n    "classificationPage1": "LLC",\r\n    "classificationPage2": "Invoice"\r\n}'

        # make sure image1 and image2 are images and not text
        try:
            Image.open(page1)
            Image.open(page2)
        except IOError:
            return {"error": "One or both of the input pages are not valid images."}

        base64_image1 = encode_image(page1)
        base64_image2 = encode_image(page2)

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
                    ],
                },
            ],
        )

        return resp

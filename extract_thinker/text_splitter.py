import instructor
from extract_thinker import Classification
from splitter import Splitter
from extract_thinker.models.doc_groups2 import DocGroups2
from typing import Any, List
from litellm import completion


class TextSplitter(Splitter):
    def __init__(self, model: str):
        self.model = model

    def belongs_to_same_document(self,
                                 obj1: Any,
                                 obj2: Any,
                                 classifications: List[Classification]
                                 ) -> DocGroups2:

        if 'image' not in obj1 or 'image' not in obj2:
            raise ValueError("Input objects must have an 'image' key")

        page1 = obj1['text']
        page2 = obj2['text']

        assistantPrompt = 'What you are an API that extracts information. You receive as input: \r\n1. two pages \r\n2. a group of classifications\r\n output:\r\nA JSON with the classification of each document and if belongs to the same document\r\n\r\n//Example 1\r\n//can be null if belongsToSamePage is true\r\n{\r\n    "belongs_to_same_document": true,\r\n    "classification_page1": "LLC",\r\n    "classification_page2": "LLC"\r\n}\r\n//Example 2\r\n{\r\n    "belongs_to_same_document": false,\r\n    "classification_page1": "LLC",\r\n    "classification_page2": "Invoice"\r\n}'
 
        classifications_text = (
            "##Classifications\n"
            + "\n".join([f"{c.name}: {c.description}" for c in classifications])
            + "\n\n##JSON Output\n"
        )

        content_pages = "##Content of the pages\n\nPage1:\n" + page1 + "Page2:\n" + page2

        client = instructor.from_litellm(completion)

        resp = client.chat.completions.create(
            model=self.model,
            max_tokens=500,
            messages=[
                {
                    "role": "system",
                    "content": assistantPrompt,
                },
                {
                    "role": "user",
                    "content": content_pages + "\n\n" + classifications_text,
                }
            ],
            response_model=DocGroups2,
        )

        return resp

from splitter import Splitter


from typing import IO, Union


class TextSplitter(Splitter):
    def belongs_to_same_document(self,
                                 page1: Union[str, IO],
                                 page2: Union[str, IO]) -> bool:
        # assistantPrompt = 'What you are an API that extracts information. You receive as input: \r\n1. two pages \r\n2. a group of classifications\r\n output:\r\nA JSON with the classification of each document and if belongs to the same document\r\n\r\n//Example 1\r\n//can be null if belongsToSamePage is true\r\n{\r\n    "belongsToSameDocument": true,\r\n    "classificationPage1": "LLC",\r\n    "classificationPage2": "LLC"\r\n}\r\n//Example 2\r\n{\r\n    "belongsToSameDocument": false,\r\n    "classificationPage1": "LLC",\r\n    "classificationPage2": "Invoice"\r\n}'

        # classifications_text = (
        #     "##Classifications\n"
        #     + "\n".join([f"{c.name}: {c.description}" for c in classifications])
        #     + "\n\n##JSON Output\n"
        # )

        # resp = litellm.completion(
        #     model="claude-3-sonnet-20240229",
        #     messages=[
        #         {
        #             "role": "system",
        #             "content": assistantPrompt,
        #         },
        #         {
        #             "role": "user",
        #             "content": [
        #                 {"type": "text", "text": classifications_text},
        #                 {"type": "text", "text": page1},
        #                 {"type": "text", "text": page2},
        #             ],
        #         },
        #     ],
        # )

        # return resp
        pass
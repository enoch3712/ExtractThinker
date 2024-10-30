import json
import instructor
from extract_thinker import Classification
from extract_thinker.models.doc_groups2 import DocGroups2
from extract_thinker.models.eager_doc_group import EagerDocGroup
from splitter import Splitter
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


    def split_lazy_doc_group(self, lazy_doc_group: DocGroups2) -> List[DocGroups2]:
        """
        Process a lazy document group to determine document boundaries
        """
        if not lazy_doc_group.belongs_to_same_document:
            # If pages don't belong together, split them into separate groups
            return [
                DocGroups2(
                    belongs_to_same_document=False,
                    classification_page1=lazy_doc_group.classification_page1,
                    classification_page2=None
                ),
                DocGroups2(
                    belongs_to_same_document=False,
                    classification_page1=lazy_doc_group.classification_page2,
                    classification_page2=None
                )
            ]
        return [lazy_doc_group]

    def split_eager_doc_group(self, document: List[dict]) -> EagerDocGroup:
        """
        Process entire document at once using eager strategy
        """
        # Combine all text from the document
        full_text = "\n".join(page['text'] for page in document)
        
        prompt = f"""##Document Text:
{full_text}

##Instructions:
1. First, analyze the document and identify logical sections
2. For each section, determine which pages belong to it based on content continuity
3. Explain your reasoning for each section
4. Return the results in JSON format

##Output Format:
{{
    "reasoning": "explanation of why these pages belong together",
    "groupOfDocuments": [
        {{
            "reasoning": "explanation of why these pages belong together",
            "pages": [1, 2]
        }}
    ]
}}

Please think through this step-by-step and provide your analysis.
Only return the JSON and nothing else.
"""

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1000,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You are a document processor that identifies logical sections in a document using careful analysis."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        try:
            result = json.loads(response.content)
            eager_group = EagerDocGroup()
            # Flatten all pages from all groups into a single list
            all_pages = [
                page 
                for group in result["groupOfDocuments"] 
                for page in group["pages"]
            ]
            eager_group.pages = sorted(all_pages)  # Ensure pages are in order
            eager_group.reasoning = result["reasoning"]
            return eager_group
        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            # Fallback: treat all pages as one group
            eager_group = EagerDocGroup()
            eager_group.pages = list(range(1, len(document) + 1))
            eager_group.reasoning = f"Fallback grouping due to error: {str(e)}"
            return eager_group
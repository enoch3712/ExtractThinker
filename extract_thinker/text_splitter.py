import json
import instructor
from typing import List, Any
from extract_thinker.models.classification import Classification
from extract_thinker.models.doc_group import DocGroups
from extract_thinker.models.doc_groups2 import DocGroups2
from extract_thinker.models.eager_doc_group import DocGroupsEager, EagerDocGroup
from extract_thinker.splitter import Splitter
from litellm import completion

class TextSplitter(Splitter):

    def __init__(self, model: str):
        self.model = model
        self.client = instructor.from_litellm(completion, mode=instructor.Mode.MD_JSON)

    def belongs_to_same_document(self,
                             obj1: Any,
                             obj2: Any,
                             classifications: List[Classification]
                             ) -> DocGroups2:
        """
        Compare two pages to determine if they belong to the same document and classify them.
        
        Args:
            obj1: First page object containing text
            obj2: Second page object containing text
            classifications: List of possible document classifications
        
        Returns:
            DocGroups2 object containing the comparison results
        """
        if 'text' not in obj1 or 'text' not in obj2:
            raise ValueError("Input objects must have a 'text' key")

        page1 = obj1['text']
        page2 = obj2['text']

        content = f"""Analyze these two pages and determine if they belong to the same document.
    Consider:
    - Content flow and continuity
    - Header/footer patterns
    - Page numbering
    - Document identifiers
    - Writing style consistency

    {self._classifications_to_text(classifications)}

    Return your analysis in the following JSON format:
    {{
        "belongs_to_same_document": true/false,
        "classification_page1": "classification name from the list above",
        "classification_page2": "classification name from the list above",
        "reasoning": "explanation of your decision"
    }}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": f"Page 1:\n{page1}\n\nPage 2:\n{page2}\n\n{content}"
                    }
                ],
                response_model=DocGroups2
            )
            return response
        except Exception as e:
            # Fallback response if analysis fails
            return DocGroups2(
                    belongs_to_same_document=True,  # Conservative approach: keep pages together
                    classification_page1=classifications[0].name,  # Default to first classification
                    classification_page2=classifications[0].name
                )
        
    def split_lazy_doc_group(self, document: List[dict], classifications: List[Classification]) -> DocGroups:
        """
        Process a document lazily by comparing consecutive pages to determine document boundaries.
        Returns a list of DocGroups2 objects representing the document groupings.
        """
        if len(document) < 2:
            # Handle single-page documents
            return [DocGroups2(
                belongs_to_same_document=True,
                classification_page1=classifications[0].name,  # Default to first classification
                classification_page2=None
            )]

        # Create and process page pairs
        page_pairs = self.split_document_into_groups(document)
        
        # Process each pair of pages
        doc_groups = []
        for page1, page2 in page_pairs:
            # Compare pages using belongs_to_same_document
            group_result = self.belongs_to_same_document(
                obj1={"text": page1['content']},
                obj2={"text": page2['content']},
                classifications=classifications
            )
            doc_groups.append(group_result)

        return self.aggregate_doc_groups(doc_groups)

    def split_eager_doc_group(self, document: List[dict], classifications: List[Classification]) -> DocGroups:
        """
        Process entire document at once using eager strategy
        """
        # Combine all text from the document
        all_texts = [page['content'] for page in document]
                
        content = f"""Analyze these text pages and determine if they belong to the same document.
Consider content flow, writing style, formatting patterns, and document structure.

{self._classifications_to_text(classifications)}

Return your analysis in the following JSON format:
    {{
        "reasoning": "detailed explanation of your analysis",
        "groupOfDocuments": [
            {{
                "classification": "Invoice",
                "pages": [1, 2]
            }}
        ]
    }}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": "\n=== PAGE BREAK ===\n".join(all_texts) + "\n\n" + content
                    }
                ],
                response_model=DocGroupsEager
            )

            # Convert DocGroupsEager into List[EagerDocGroup]
            eager_groups = []
            for group in response.groupOfDocuments:
                eager_group = EagerDocGroup(
                    pages=group.pages,
                    classification=group.classification
                )
                eager_groups.append(eager_group)
                
            return eager_groups
            
        except Exception as e:
            # Fallback: treat all pages as one group
            eager_group = EagerDocGroup(
                pages=list(range(1, len(document) + 1)),
                classification="unknown"  # Add a default classification
            )
            return [eager_group]
        
    def _classifications_to_text(self, classifications: List[Classification]) -> str:
        """
        Converts a list of Classification objects into a formatted text string
        including their names, descriptions and contract structures.
        """
        classifications_text = "##Classifications\n"
        for classification in classifications:
            classifications_text += f"### {classification.name}\n"
            classifications_text += f"**Description:** {classification.description}\n\n"
            
            if classification.contract:
                classifications_text += self._add_classification_structure(classification)

        return classifications_text
    
    def _add_classification_structure(self, classification: Classification) -> str:
        content = ""
        if classification.contract:
            content = "\t##Contract Structure:\n"
            for name, field in classification.contract.model_fields.items():
                field_str = str(field)
                field_type = field_str.split('=')[1].split(' ')[0]
                required = 'required' in field_str
                attributes = f"required={required}"
                field_details = f"\t\tName: {name}, Type: {field_type}, Attributes: {attributes}"
                content += field_details + "\n"
        return content
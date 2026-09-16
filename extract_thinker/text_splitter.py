from typing import List, Any
from extract_thinker.models.classification import Classification
from extract_thinker.models.doc_group import DocGroups
from extract_thinker.models.doc_groups2 import DocGroups2
from extract_thinker.splitter import Splitter
from extract_thinker.llm import LLM
from extract_thinker.models.split_classification import NumericPagePair, NumericDocumentGroups

class TextSplitter(Splitter):

    def __init__(self, model: str):
        self.model = model
        self.llm = LLM(model)

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

    Use the numeric classification IDs from the list, never names.
Include every source page exactly once, in original order.
Return your analysis in the following JSON format:
    {{
        "belongs_to_same_document": true/false,
        "classification_page1": 1,
        "classification_page2": 1,
        "reasoning": "explanation of your decision"
    }}"""

        response = self.llm.request(
            messages=[
                {
                    "role": "user",
                    "content": f"Page 1:\n{page1}\n\nPage 2:\n{page2}\n\n{content}"
                }
            ],
            response_model=NumericPagePair
        )
        return self._resolve_pair(response, classifications)

    def split_lazy_doc_group(self, document: List[dict], classifications: List[Classification]) -> DocGroups:
        """
        Process a document lazily by comparing consecutive pages to determine document boundaries.
        Returns a list of DocGroups2 objects representing the document groupings.
        """
        if len(document) < 2:
            from extract_thinker.models.doc_group import DocGroup
            result = DocGroups()
            result.doc_groups = [DocGroup(group.pages, group.classification, group.classification_id)
                                 for group in self.split_eager_doc_group(document, classifications)]
            return result

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
        if not document:
            return []
        # Combine all text from the document
        all_texts = [page['content'] for page in document]
                
        content = f"""Analyze these text pages and determine if they belong to the same document.
Consider content flow, writing style, formatting patterns, and document structure.

{self._classifications_to_text(classifications)}

Use the numeric classification IDs from the list, never names.
Include every source page exactly once, in original order.
Return your analysis in the following JSON format:
    {{
        "reasoning": "detailed explanation of your analysis",
        "groupOfDocuments": [
            {{
                "classification": 1,
                "pages": [1, 2]
            }}
        ]
    }}"""
        
        response = self.llm.request(
            messages=[
                {
                    "role": "user",
                    "content": "\n=== PAGE BREAK ===\n".join(all_texts) + "\n\n" + content
                }
            ],
            response_model=NumericDocumentGroups
        )

        return self._resolve_eager(response, classifications, len(document))

    def _classifications_to_text(self, classifications: List[Classification]) -> str:
        """
        Converts a list of Classification objects into a formatted text string
        including their names, descriptions and contract structures.
        """
        if not classifications:
            raise ValueError("At least one classification is required")
        classifications_text = "##Classifications (numeric IDs)\n"
        for identifier, classification in enumerate(classifications, 1):
            classifications_text += f"### ID {identifier}: {classification.name}\n"
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
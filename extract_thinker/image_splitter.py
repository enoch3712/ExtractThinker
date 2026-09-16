import base64
from io import BytesIO
from typing import List, Any
from extract_thinker.models.classification import Classification
from extract_thinker.models.doc_group import DocGroups
from extract_thinker.models.doc_groups2 import DocGroups2
from extract_thinker.splitter import Splitter
from extract_thinker.llm import LLM
from extract_thinker.models.split_classification import NumericPagePair, NumericDocumentGroups

class ImageSplitter(Splitter):

    def __init__(self, model: str):
        self.model = model
        self.llm = LLM(model)

    def encode_image(self, image):
        """
        Encode an image to base64 string.
        
        Args:
            image: Either a PIL Image object or bytes of an image
            
        Returns:
            str: Base64 encoded image string
        """
        from extract_thinker.utils import encode_image
        return encode_image(image)

    @staticmethod
    def _data_url(encoded):
        from PIL import Image
        with Image.open(BytesIO(base64.b64decode(encoded))) as image:
            mime = Image.MIME.get(image.format, "image/png")
        return f"data:{mime};base64,{encoded}"

    def belongs_to_same_document(self,
                             obj1: Any,
                             obj2: Any,
                             classifications: List[Classification]
                             ) -> DocGroups2:
        """
        Compare two pages to determine if they belong to the same document and classify them.
        
        Args:
            obj1: First page object containing an image
            obj2: Second page object containing an image
            classifications: List of possible document classifications
        
        Returns:
            DocGroups2 object containing the comparison results
        """
        if 'image' not in obj1 or 'image' not in obj2:
            raise ValueError("Input objects must have an 'image' key")

        page1 = obj1['image']
        page2 = obj2['image']

        # Encode images to base64
        base64_image1 = self.encode_image(page1)
        base64_image2 = self.encode_image(page2)

        content = f"""Analyze these two pages and determine if they belong to the same document.
    Consider:
    - Visual consistency and layout
    - Content flow and continuity
    - Header/footer patterns
    - Page numbering
    - Document identifiers

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

        # Add all images to the content
        messages = [{"type": "text", "text": content}]
        messages.append({
            "type": "image_url",
            "image_url": {
                "url": self._data_url(base64_image1)
            }
        })
        messages.append({
            "type": "image_url",
            "image_url": {
                "url": self._data_url(base64_image2)
            }
        })

        response = self.llm.request(
            messages=[
                {
                    "role": "user",
                    "content": messages
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
                obj1={"image": page1["image"]},
                obj2={"image": page2["image"]},
                classifications=classifications
            )
            doc_groups.append(group_result)

        return self.aggregate_doc_groups(doc_groups)

    def split_eager_doc_group(self, document: List[dict], classifications: List[Classification]) -> DocGroups:
        """
        Process entire document of images at once using eager strategy
        """
        if not document:
            return []
        # Encode all images
        base64_images = [self.encode_image(page['image']) for page in document]
                
        content = f"""Analyze these images and determine if they belong to the same document.
Consider visual consistency, layout, content flow, and any header/footer patterns.

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
        
        messages = [{"type": "text", "text": content}]

        # Add classification example images if they exist
        for identifier, classification in enumerate(classifications, 1):
            if classification.image:
                messages.append({
                    "type": "text",
                    "text": f"Example for classification ID {identifier}: {classification.name}\n"
                })
                messages.append({
                    "type": "image_url",
                    "image_url": {
                        "url": self._data_url(self.encode_image(classification.image))
                    }
                })
            
        # Add all images to the content
        for base64_image in base64_images:
            messages.append({
                "type": "image_url",
                "image_url": {
                    "url": self._data_url(base64_image)
                }
            })
        
        response = self.llm.request(
            messages=[
                {
                    "role": "user",
                    "content": messages
                }
            ],
            response_model=NumericDocumentGroups
        )

        return self._resolve_eager(response, classifications, len(document))

    def _classifications_to_text(self, classifications: List[Classification]) -> str:
        """
        Converts a list of Classification objects into a formatted text string
        including their names, descriptions, contract structures, and images.
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
            # Iterate over the fields of the contract attribute if it's not None
            for name, field in classification.contract.model_fields.items():
                # Extract the type and required status from the field's string representation
                field_str = str(field)
                field_type = field_str.split('=')[1].split(' ')[0]  # Extracts the type
                required = 'required' in field_str  # Checks if 'required' is in the string
                # Creating a string representation of the field attributes
                attributes = f"required={required}"
                # Append each field's details to the content string
                field_details = f"\t\tName: {name}, Type: {field_type}, Attributes: {attributes}"
                content += field_details + "\n"
        return content
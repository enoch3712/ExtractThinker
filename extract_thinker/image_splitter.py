import base64
import litellm
import instructor
from io import BytesIO
from typing import List, Any
from extract_thinker.models.classification import Classification
from extract_thinker.models.doc_group import DocGroups
from extract_thinker.models.doc_groups2 import DocGroups2
from extract_thinker.models.eager_doc_group import DocGroupsEager, EagerDocGroup
from extract_thinker.splitter import Splitter

class ImageSplitter(Splitter):

    def __init__(self, model: str):
        if not litellm.supports_vision(model=model):
            raise ValueError(f"Model {model} is not supported for ImageSplitter, since its not a vision model.")
        self.model = model
        self.client = instructor.from_litellm(litellm.completion, mode=instructor.Mode.MD_JSON)

    def encode_image(self, image):
        """
        Encode an image to base64 string.
        
        Args:
            image: Either a PIL Image object or bytes of an image
            
        Returns:
            str: Base64 encoded image string
        """
        if isinstance(image, bytes):
            # If already bytes, encode directly
            return base64.b64encode(image).decode("utf-8")
        else:
            # If PIL Image, convert to bytes first
            buffered = BytesIO()
            image.save(buffered, format=image.format or 'JPEG')
            img_byte = buffered.getvalue()
            return base64.b64encode(img_byte).decode("utf-8")

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

    Return your analysis in the following JSON format:
    {{
        "belongs_to_same_document": true/false,
        "classification_page1": "classification name from the list above",
        "classification_page2": "classification name from the list above",
        "reasoning": "explanation of your decision"
    }}"""

        # Add all images to the content
        messages = [{"type": "text", "text": content}]
        messages.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image1}"
            }
        })
        messages.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image2}"
            }
        })

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": messages
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
        # Encode all images
        base64_images = [self.encode_image(page['image']) for page in document]
                
        content = f"""Analyze these images and determine if they belong to the same document.
Consider visual consistency, layout, content flow, and any header/footer patterns.

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
        
        # messages = [{"type": "text", "text": content}]

        # Add classification example images if they exist
        for classification in classifications:
            if classification.image:
                messages.append({
                    "type": "text",
                    "text": f"## {classification.name} Classification\n"
                })
                messages.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self.encode_image(classification.image)}"
                    }
                })
            
        # Add all images to the content
        messages = [{"type": "text", "text": content}]
        for base64_image in base64_images:
            messages.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": messages
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
            #TODO
            # Fallback: treat all images as one group
            eager_group = EagerDocGroup(
                pages=list(range(1, len(document) + 1)),
                classification="unknown"  # Add a default classification
            )
            return [eager_group]
        
    def _classifications_to_text(self, classifications: List[Classification]) -> str:
        """
        Converts a list of Classification objects into a formatted text string
        including their names, descriptions, contract structures, and images.
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
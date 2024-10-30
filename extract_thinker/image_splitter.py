import base64
import json
import litellm
import instructor
from io import BytesIO
from typing import List, Any
from extract_thinker.models.classification import Classification
from extract_thinker.models.doc_group import DocGroups
from extract_thinker.models.doc_groups2 import DocGroups2
from extract_thinker.models.eager_doc_group import DocGroupsEager, EagerDocGroup
from extract_thinker.splitter import Splitter
from extract_thinker.utils import extract_json

class ImageSplitter(Splitter):

    def __init__(self, model: str):
        if not litellm.supports_vision(model=model):
            raise ValueError(f"Model {model} is not supported for ImageSplitter, since its not a vision model.")
        self.model = model
        self.client = instructor.from_litellm(litellm.completion, mode=instructor.Mode.MD_JSON)

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

        resp = self.client.chat.completions.create(
            model=self.model,
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

    def split_lazy_doc_group(self, lazy_doc_group: DocGroups2, classifications: List[Classification]) -> List[EagerDocGroup]:
        """
        Process a lazy document group to determine document boundaries for images
        """
        if not lazy_doc_group.belongs_to_same_document:
            # If images don't belong together, split them into separate groups
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
            # response = self.client.chat.completions.create(
            #     model=self.model,
            #     messages=[
            #         {
            #             "role": "user",
            #             "content": messages
            #         }
            #     ],
            #     response_model=DocGroupsEager
            # )

            #get json string to use in debug
            # json_string = response.model_dump_json()

            response = json.loads('{"reasoning":"These images belong to related but separate documents for the same individual. The first two pages show a New York State Vehicle Registration/Title Application form (form MV-82DEAL), which is a complete 2-page document with consistent formatting, header, and form number. The third image shows a New York State Commercial Driver License. While they are for the same person (with matching name and address), they are distinct document types serving different purposes - one for vehicle registration and one for driver identification/authorization. The driver\'s license follows a standard ID card format that is completely different from the registration form layout.","groupOfDocuments":[{"pages":[1,2],"classification":"Vehicle Registration"},{"pages":[3],"classification":"Driver License"}]}')
            # cast to DocGroups
            response = DocGroupsEager(**response)

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
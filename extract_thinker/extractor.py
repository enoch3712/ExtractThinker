import asyncio
import base64
from io import BytesIO
from typing import Any, Dict, List, Optional, IO, Union

from pydantic import BaseModel
from extract_thinker.document_loader.document_loader import DocumentLoader
from extract_thinker.models.classification import Classification
from extract_thinker.models.classification_response import (
    ClassificationResponse,
)
from extract_thinker.llm import LLM
import os

from extract_thinker.document_loader.loader_interceptor import LoaderInterceptor
from extract_thinker.document_loader.llm_interceptor import LlmInterceptor

from extract_thinker.utils import get_file_extension, encode_image, json_to_formatted_string
import yaml
import litellm

SUPPORTED_IMAGE_FORMATS = ["jpeg", "png", "bmp", "tiff"]
SUPPORTED_EXCEL_FORMATS = ['.xls', '.xlsx', '.xlsm', '.xlsb', '.odf', '.ods', '.odt', '.csv']


class Extractor:
    def __init__(
        self, processor: Optional[DocumentLoader] = None, llm: Optional[LLM] = None
    ):
        self.document_loader: Optional[DocumentLoader] = processor
        self.llm: Optional[LLM] = llm
        self.file: Optional[str] = None
        self.document_loaders_by_file_type: Dict[str, DocumentLoader] = {}
        self.loader_interceptors: List[LoaderInterceptor] = []
        self.llm_interceptors: List[LlmInterceptor] = []
        self.is_classify_image: bool = False

    def add_interceptor(
        self, interceptor: Union[LoaderInterceptor, LlmInterceptor]
    ) -> None:
        if isinstance(interceptor, LoaderInterceptor):
            self.loader_interceptors.append(interceptor)
        elif isinstance(interceptor, LlmInterceptor):
            self.llm_interceptors.append(interceptor)
        else:
            raise ValueError(
                "Interceptor must be an instance of LoaderInterceptor or LlmInterceptor"
            )

    def set_document_loader_for_file_type(
        self, file_type: str, document_loader: DocumentLoader
    ):
        self.document_loaders_by_file_type[file_type] = document_loader

    def get_document_loader_for_file(self, file: str) -> DocumentLoader:
        _, ext = os.path.splitext(file)
        return self.document_loaders_by_file_type.get(ext, self.document_loader)

    def load_document_loader(self, document_loader: DocumentLoader) -> None:
        self.document_loader = document_loader

    def load_llm(self, model: Optional[str] = None) -> None:
        if isinstance(model, LLM):
            self.llm = model
        elif model is not None:
            self.llm = LLM(model)
        else:
            raise ValueError("Either a model string or an LLM object must be provided.")

    def extract(self, source: Union[str, IO, list], response_model: type[BaseModel], vision: bool = False, content: Optional[str] = None) -> Any:
        self.extra_content = content

        if not issubclass(response_model, BaseModel):
            raise ValueError("response_model must be a subclass of Pydantic's BaseModel.")

        if isinstance(source, str):  # if it's a file path
            return self.extract_from_file(source, response_model, vision)
        elif isinstance(source, IO):  # if it's a stream
            return self.extract_from_stream(source, response_model, vision)
        elif isinstance(source, list) and all(isinstance(item, dict) for item in source):  # if it's a list of dictionaries
            return self.extract_from_list(source, response_model, vision)
        else:
            raise ValueError("Source must be a file path, a stream, or a list of dictionaries")

    async def extract_async(self, source: Union[str, IO, list], response_model: type[BaseModel], vision: bool = False) -> Any:
        return await asyncio.to_thread(self.extract, source, response_model, vision)

    def extract_from_list(self, data: List[Dict[Any, Any]], response_model: type[BaseModel], vision: bool) -> str:
        # check if document_loader is None, raise error
        if self.document_loader is None:
            raise ValueError("Document loader is not set")

        content = "\n".join([f"#{k}:\n{v}" for d in data for k, v in d.items() if k != "image"])
        return self._extract(content, data, response_model, vision, is_stream=False)

    def extract_from_file(
        self, file: str, response_model: str, vision: bool = False
    ) -> str:
        if self.document_loader is not None:
            content = self.document_loader.load_content_from_file(file)
        else:
            document_loader = self.get_document_loader_for_file(file)
            if document_loader is None:
                raise ValueError("No suitable document loader found for file type")
            content = document_loader.load_content_from_file(file)
        return self._extract(content, file, response_model, vision)

    def extract_from_stream(
        self, stream: IO, response_model: str, vision: bool = False
    ) -> str:
        # check if document_loader is None, raise error
        if self.document_loader is None:
            raise ValueError("Document loader is not set")

        content = self.document_loader.load(stream)
        return self._extract(content, stream, response_model, vision, is_stream=True)

    def classify_from_image(self, image: Any, classifications: List[Classification]):
        # requires no content extraction from loader
        content = {
            "image": image,
        }
        return self._classify(content, classifications, image)

    def classify_from_path(self, path: str, classifications: List[Classification]):
        content = self.document_loader.load_content_from_file_list(path) if self.is_classify_image else self.document_loader.load_content_from_file(path)
        return self._classify(content, classifications)

    def classify_from_stream(self, stream: IO, classifications: List[Classification]):
        content = self.document_loader.load_content_from_stream_list(stream) if self.is_classify_image else self.document_loader.load_content_from_stream(stream)
        self._classify(content, classifications)

    def classify_from_excel(self, path: Union[str, IO], classifications: List[Classification]):
        if isinstance(path, str):
            content = self.document_loader.load_content_from_file(path)
        else:
            content = self.document_loader.load_content_from_stream(path)
        return self._classify(content, classifications)

    # def classify_with_image(self, messages: List[Dict[str, Any]]):
    #     resp = litellm.completion(self.llm.model, messages)

    #     return ClassificationResponse(**resp.choices[0].message.content)

    def _add_classification_structure(self, classification: Classification) -> str:
        content = ""
        if classification.contract:
            content = "\tContract Structure:\n"
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

    def _classify(self, content: Any, classifications: List[Classification], image: Optional[Any] = None):
        messages = [
            {
                "role": "system",
                "content": "You are a server API that receives document information "
                "and returns specific fields in JSON format.\n",
            },
        ]

        if self.is_classify_image:
            input_data = (
                f"##Take the first image, and compare to the several images provided. Then classificationaccording to the classifcation attached to the image\n"
                + "Output Example: \n"
                + "{\r\n\t\"name\": \"DMV Form\",\r\n\t\"confidence\": 8\r\n}"
                + "\n\n##ClassificationResponse JSON Output\n"
            )

        else:
            input_data = (
                f"##Content\n{content}\n##Classifications\n#if contract present, each field present increase confidence level\n"
                + "\n".join([f"{c.name}: {c.description} \n{self._add_classification_structure(c)}" for c in classifications])
                + "#Dont use contract structure, just to help on the ClassificationResponse\nOutput Example: \n"
                + "{\r\n\t\"name\": \"DMV Form\",\r\n\t\"confidence\": 8\r\n}"
                + "\n\n##ClassificationResponse JSON Output\n"
            )

        #messages.append({"role": "user", "content": input_data})

        if self.is_classify_image:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": input_data,
                        },
                    ],
                }
            )
            for classification in classifications:
                if classification.image:
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "{classification.name}: {classification.description}"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64," + encode_image(classification.image)
                                },
                            },
                        ],
                    })
                else:
                    raise ValueError(f"Image required for classification '{classification.name}' but not found.")

            response = self.llm.request(messages, ClassificationResponse)
        else:
            messages.append({"role": "user", "content": input_data})
            response = self.llm.request(messages, ClassificationResponse)

        return response

    def classify(self, input: Union[str, IO], classifications: List[Classification], image: bool = False):
        self.is_classify_image = image

        if image:
            return self.classify_from_image(input, classifications)

        if isinstance(input, str):
            # Check if the input is a valid file path
            if os.path.isfile(input):
                file_type = get_file_extension(input)
                if file_type == 'pdf':
                    return self.classify_from_path(input, classifications)
                elif file_type in SUPPORTED_EXCEL_FORMATS:
                    return self.classify_from_excel(input, classifications)
                else:
                    raise ValueError(f"Unsupported file type: {input}")
            else:
                raise ValueError(f"No such file: {input}")
        elif hasattr(input, 'read'):
            # Check if the input is a stream (like a file object)
            return self.classify_from_stream(input, classifications)
        else:
            raise ValueError("Input must be a file path or a stream.")

    async def classify_async(self, input: Union[str, IO], classifications: List[Classification]):
        return await asyncio.to_thread(self.classify, input, classifications)

    def _extract(self,
                 content,
                 file_or_stream,
                 response_model,
                 vision=False,
                 is_stream=False
                 ):
        # call all the llm interceptors before calling the llm
        for interceptor in self.llm_interceptors:
            interceptor.intercept(self.llm)

        messages = [
            {
                "role": "system",
                "content": "You are a server API that receives document information "
                "and returns specific fields in JSON format.",
            },
        ]

        if self.extra_content is not None:
            if isinstance(self.extra_content, dict):
                self.extra_content = yaml.dump(self.extra_content)
            messages.append({"role": "user", "content": "##Extra Content\n\n" + self.extra_content})

        if content is not None:
            if isinstance(content, dict):
                if content["is_spreadsheet"]:
                    content = json_to_formatted_string(content["data"])
                content = yaml.dump(content)
            messages.append({"role": "user", "content": "##Content\n\n" + content})

        if vision:
            base64_encoded_image = encode_image(
                file_or_stream, is_stream
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64," + base64_encoded_image
                            },
                        },
                    ],
                }
            ]

        response = self.llm.request(messages, response_model)
        return response

    def loadfile(self, file):
        self.file = file
        return self

    def loadstream(self, stream):
        return self

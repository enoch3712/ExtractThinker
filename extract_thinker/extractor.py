import asyncio
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

from extract_thinker.utils import get_file_extension, encode_image
import yaml


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
        self.extra_content: Optional[str] = None

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

    def classify_from_path(self, path: str, classifications: List[Classification]):
        content = self.document_loader.load_content_from_file(path)
        return self._classify(content, classifications)

    def classify_from_stream(self, stream: IO, classifications: List[Classification]):
        content = self.document_loader.load_content_from_stream(stream)
        self._classify(content, classifications)

    def classify_from_excel(self, path: Union[str, IO], classifications: List[Classification]):
        if isinstance(path, str):
            content = self.document_loader.load_content_from_file(path)
        else:
            content = self.document_loader.load_content_from_stream(path)
        return self._classify(content, classifications)

    def _classify(self, content: str, classifications: List[Classification]):
        messages = [
            {
                "role": "system",
                "content": "You are a server API that receives document information "
                "and returns specific fields in JSON format.",
            },
        ]

        input_data = (
            f"##Content\n{content}\n##Classifications\n"
            + "\n".join([f"{c.name}: {c.description}" for c in classifications])
            + "\n\n##JSON Output\n"
        )

        messages.append({"role": "user", "content": input_data})

        response = self.llm.request(messages, ClassificationResponse)

        return response

    def classify(self, input: Union[str, IO], classifications: List[Classification]):
        if isinstance(input, str):
            # Check if the input is a valid file path
            if os.path.isfile(input):
                file_type = get_file_extension(input)
                if file_type in SUPPORTED_IMAGE_FORMATS:
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
                        {"type": "text", "text": "Whats in this image?"},
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

from typing import List, Optional, IO, Union
from extract_thinker.document_loader.document_loader import DocumentLoader
from extract_thinker.models import (
    Classification,
    ClassificationResponse,
)
from extract_thinker.splitter import Splitter
from extract_thinker.llm import LLM
import os

from extract_thinker.document_loader.loader_interceptor import LoaderInterceptor
from extract_thinker.document_loader.llm_interceptor import LlmInterceptor


class Extractor:
    def __init__(
        self, processor: Optional[DocumentLoader] = None, llm: Optional[LLM] = None
    ):
        self.document_loader: Optional[DocumentLoader] = processor
        self.llm: Optional[LLM] = llm
        self.splitter: Optional[Splitter] = None
        self.file: Optional[str] = None
        self.document_loaders_by_file_type = {}
        self.loader_interceptors: List[LoaderInterceptor] = []
        self.llm_interceptors: List[LlmInterceptor] = []

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

    def load_llm(self, model: str) -> None:
        self.llm = LLM(model)

    def extract(
        self, source: Union[str, IO], response_model: str, vision: bool = False
    ) -> str:
        if isinstance(source, str):  # if it's a file path
            return self.extract_from_file(source, response_model, vision)
        elif isinstance(source, IO):  # if it's a stream
            return self.extract_from_stream(source, response_model, vision)
        else:
            raise ValueError("Source must be a file path or a stream")

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

    def classify(self, input: Union[str, IO]):
        if isinstance(input, str):
            # Check if the input is a valid file path
            if os.path.isfile(input):
                _, ext = os.path.splitext(input)
                if ext.lower() == ".pdf":
                    return self.classify_from_path(input)
                else:
                    raise ValueError(f"Unsupported file type: {ext}")
            else:
                raise ValueError(f"No such file: {input}")
        elif hasattr(input, 'read'):
            # Check if the input is a stream (like a file object)
            return self.classify_from_stream(input)
        else:
            raise ValueError("Input must be a file path or a stream.")

    def _extract(
        self, content, file_or_stream, response_model, vision=False, is_stream=False
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

        if vision:
            base64_encoded_image = self._encode_image_to_base64(
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
        else:
            messages.append({"role": "user", "content": content})

        response = self.llm.request(messages, response_model)
        return response

    def loadfile(self, file):
        self.file = file
        return self

    def loadstream(self, stream):
        return self

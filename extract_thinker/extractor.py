from typing import List, Optional, IO, Union
from extract_thinker.document_loader.document_loader import DocumentLoader
from extract_thinker.models import (
    Classification,
    ClassificationResponse,
    DocGroups2,
    DocGroup,
    DocGroups,
)
from extract_thinker.splitter import Splitter
from extract_thinker.llm import LLM
import asyncio
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
        doc_groups: Optional[DocGroups] = None
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

    def extract(self, source: Union[str, IO], response_model: str, vision: bool = False) -> str:
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

        content = self.document_loader.load_content_from_stream(stream)
        return self._extract(content, stream, response_model, vision, is_stream=True)

    def classify_from_path(self, path: str, classifications: List[Classification]):
        content = self.document_loader.getContent(path)
        return self._classify(content, classifications)

    def classify_from_stream(self, stream: IO, classifications: List[Classification]):
        content = self.document_loader.getContentFromStream(stream)
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
            f"##Content\n{content[0]}\n##Classifications\n"
            + "\n".join([f"{c.name}: {c.description}" for c in classifications])
            + "\n\n##JSON Output\n"
        )

        messages.append({"role": "user", "content": input_data})

        response = self.llm.request(messages, ClassificationResponse)

        return response

    def _extract(
        self, content, file_or_stream, response_model, vision=False, is_stream=False
    ):

        #call all the llm interceptors before calling the llm
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

    def split(self, classifications: List[Classification]):
        splitter = self.splitter

        # Check if the file is a PDF
        _, ext = os.path.splitext(self.file)
        if ext.lower() != ".pdf":
            raise ValueError("Invalid file type. Only PDFs are accepted.")

        images = self.document_loader.convert_pdf_to_images(self.file)

        groups = splitter.split_document_into_groups([self.file])

        loop = asyncio.get_event_loop()
        processedGroups = loop.run_until_complete(
            splitter.process_split_groups(groups, classifications)
        )

        doc_groups = self.aggregate_split_documents_2(processedGroups)

        self.doc_groups = doc_groups

        return self

    def aggregate_split_documents_2(doc_groups_tasks: List[DocGroups2]) -> DocGroups:
        doc_groups = DocGroups()
        current_group = DocGroup()
        page_number = 1

        # do the first group outside of the loop
        doc_group = doc_groups_tasks[0]

        if doc_group.belongs_to_same_document:
            current_group.pages = [1, 2]
            current_group.classification = doc_group.classification_page1
            current_group.certainties = [
                doc_group.certainty,
                doc_groups_tasks[1].certainty,
            ]
        else:
            current_group.pages = [1]
            current_group.classification = doc_group.classification_page1
            current_group.certainties = [doc_group.certainty]

            doc_groups.doc_groups.append(current_group)

            current_group = DocGroup()
            current_group.pages = [2]
            current_group.classification = doc_group.classification_page2
            current_group.certainties = [doc_groups_tasks[1].certainty]

        page_number += 1

        for index in range(1, len(doc_groups_tasks)):
            doc_group_2 = doc_groups_tasks[index]

            if doc_group_2.belongs_to_same_document:
                current_group.pages.append(page_number + 1)
                current_group.certainties.append(doc_group_2.certainty)
            else:
                doc_groups.doc_groups.append(current_group)

                current_group = DocGroup()
                current_group.classification = doc_group_2.classification_page2
                current_group.pages = [page_number + 1]
                current_group.certainties = [doc_group_2.certainty]

            page_number += 1

        doc_groups.doc_groups.append(current_group)  # the last group

        return doc_groups

    def where(self, condition):
        return self

    def loadfile(self, file):
        self.file = file
        return self

    def loadstream(self, stream):
        return self

    def loadSplitter(self, splitter):
        self.splitter = splitter
        return self

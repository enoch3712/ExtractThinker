import json
import os
import re
import mimetypes
from typing import Optional, Any, List, Union, Sequence
from pydantic import BaseModel, Field
from io import BytesIO
from operator import attrgetter
from cachetools import cachedmethod
from cachetools.keys import hashkey
from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1 as documentai
from google.oauth2 import service_account

from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader

class Config(BaseModel):
    enable_native_pdf_parsing: bool = Field(
        default=False, description="Enable native PDF parsing"
    )
    page_range: Optional[List[int]] = Field(
        default=None, description="The page range to process"
    )

class DocumentLoaderDocumentAI(CachedDocumentLoader):
    PROCESSOR_NAME_PATTERN = r"projects\/[0-9]+\/locations\/[a-z\-0-9]+\/processors\/[a-z0-9]+"

    def __init__(
        self,
        credentials: str,
        location: str,
        processor_name: str,
        content: Any = None,
        cache_ttl: int = 300,
    ):
        super().__init__(content, cache_ttl)
        self._validate_processor_name(processor_name)
        self.credentials = self._parse_credentials(credentials)
        self.processor_name = processor_name
        self.location = location
        self.client = self._create_client()

    @staticmethod
    def _validate_processor_name(processor_name: str) -> None:
        if not re.fullmatch(DocumentLoaderDocumentAI.PROCESSOR_NAME_PATTERN, processor_name):
            raise ValueError(
                f"Processor name {processor_name} has the wrong format. It should be in the format of "
                "projects/PROJECT_ID/locations/{LOCATION}/processors/PROCESSOR_ID"
            )

    @staticmethod
    def _parse_credentials(credentials: str) -> service_account.Credentials:
        if credentials is None:
            raise ValueError("Credentials cannot be None")

        try:
            cred_dict = json.loads(credentials)
            return service_account.Credentials.from_service_account_info(cred_dict)
        except json.JSONDecodeError:
            if os.path.isfile(credentials):
                return service_account.Credentials.from_service_account_file(credentials)
            else:
                raise ValueError("Invalid credentials: must be a JSON string or a path to a JSON file")

    def _create_client(self) -> documentai.DocumentProcessorServiceClient:
        return documentai.DocumentProcessorServiceClient(
            credentials=self.credentials,
            client_options=ClientOptions(
                api_endpoint=f"{self.location}-documentai.googleapis.com"
            ),
        )

    @staticmethod
    def _resolve_mime_type(file_path: str) -> str:
        return mimetypes.guess_type(file_path)[0]

    @cachedmethod(
        cache=attrgetter("cache"), key=lambda self, file_path: hashkey(file_path)
    )
    def load_content_from_file(
        self, file_path: str, config: Optional[Config] = None
    ) -> dict:
        config = config or Config()
        try:
            with open(file_path, "rb") as document:
                document_content = document.read()
                return self._process_document(document_content, self._resolve_mime_type(file_path), config)
        except Exception as e:
            raise Exception(f"Error processing file: {e}") from e

    @cachedmethod(
        cache=attrgetter("cache"), key=lambda self, stream, mime_type: hashkey(id(stream), mime_type)
    )
    def load_content_from_stream(
        self,
        stream: Union[BytesIO, str],
        mime_type: str,
        config: Optional[Config] = None
    ) -> dict:
        config = config or Config()
        try:
            return self._process_document(stream.read(), mime_type, config)
        except Exception as e:
            raise Exception(f"Error processing stream: {e}") from e

    def _process_document(self, content: bytes, mime_type: str, config: Config) -> dict:
        response = self.client.process_document(
            request=documentai.ProcessRequest(
                name=self.processor_name,
                raw_document=documentai.RawDocument(
                    content=content,
                    mime_type=mime_type,
                ),
                process_options=self._create_process_options(config),
                skip_human_review=True,
            ),
        )
        return self._process_result(response)

    @staticmethod
    def _create_process_options(config: Config) -> documentai.ProcessOptions:
        return documentai.ProcessOptions(
            ocr_config=documentai.OcrConfig(
                enable_native_pdf_parsing=config.enable_native_pdf_parsing,
            ),
            individual_page_selector=(
                documentai.IndividualPageSelector(page_range=config.page_range)
                if config.page_range else None
            ),
        )

    def _process_result(self, result: documentai.ProcessResponse) -> dict:
        return {
            "pages": [
                self._process_page(result.document.text, page)
                for page in result.document.pages
            ]
        }

    def _process_page(self, full_text: str, page: documentai.Document.Page) -> dict:
        return {
            "content": self._get_page_full_content(full_text, page),
            "paragraphs": self._get_page_paragraphs(full_text, page),
            "tables": self._get_page_tables(full_text, page),
        }

    @staticmethod
    def _get_page_full_content(full_text: str, page: documentai.Document.Page) -> str:
        start_index = page.tokens[0].layout.text_anchor.text_segments[0].start_index
        end_index = page.tokens[-1].layout.text_anchor.text_segments[-1].end_index
        return full_text[start_index:end_index]

    @staticmethod
    def _get_page_paragraphs(full_text: str, page: documentai.Document.Page) -> List[str]:
        return [
            full_text[paragraph.layout.text_anchor.text_segments[0].start_index:
                      paragraph.layout.text_anchor.text_segments[-1].end_index]
            for paragraph in page.paragraphs
        ]

    def _get_page_tables(self, full_text: str, page: documentai.Document.Page) -> List[List[str]]:
        return [
            self._get_table_data(full_text, table.header_rows) +
            self._get_table_data(full_text, table.body_rows)
            for table in page.tables
        ]

    @staticmethod
    def _get_table_data(full_text: str, rows: Sequence[documentai.Document.Page.Table.TableRow]) -> List[List[str]]:
        return [
            [
                full_text[cell.layout.text_anchor.text_segments[0].start_index:
                          cell.layout.text_anchor.text_segments[-1].end_index].strip()
                for cell in row.cells
            ]
            for row in rows
        ]

    def load_content_from_file_list(self, file_paths: List[str]) -> List[dict]:
        return [self.load_content_from_file(file_path) for file_path in file_paths]

    def load_content_from_stream_list(self, streams: List[BytesIO]) -> List[dict]:
        return [self.load_content_from_stream(stream) for stream in streams]

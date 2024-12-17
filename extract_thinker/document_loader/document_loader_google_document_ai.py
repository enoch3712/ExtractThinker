import json
import os
import re
import mimetypes
from typing import Optional, Any, Dict, List, Union, Sequence
from pydantic import BaseModel, Field
from io import BytesIO
from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1 as documentai
from google.oauth2 import service_account

from extract_thinker.document_loader.document_loader import DocumentLoader


class Config(BaseModel):
    enable_native_pdf_parsing: bool = Field(
        default=False, description="Enable native PDF parsing"
    )
    page_range: Optional[List[int]] = Field(
        default=None, description="The page range to process"
    )


class DocumentLoaderDocumentAI(DocumentLoader):
    """Loader for documents using Google Document AI."""
    
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

    def load(self, source: Union[str, BytesIO], config: Optional[Config] = None) -> List[Dict[str, Any]]:
        """
        Load and analyze a document using Google Document AI.
        Returns a list of pages, each containing:
        - content: The text content of the page
        - tables: Any tables found on the page
        - image: The page image (if vision_mode is True)
        
        Args:
            source: Either a file path or BytesIO stream
            config: Optional configuration for Document AI processing
            
        Returns:
            List[Dict[str, Any]]: List of pages with content and optional images
        """
        if not self.can_handle(source):
            raise ValueError(f"Cannot handle source: {source}")

        config = config or Config()
        try:
            # Get document content and mime type
            if isinstance(source, str):
                with open(source, "rb") as document:
                    document_content = document.read()
                mime_type = self._resolve_mime_type(source)
            else:
                document_content = source.read()
                mime_type = mimetypes.guess_type(source.name)[0] if hasattr(source, 'name') else 'application/pdf'

            # Process with Document AI
            response = self.client.process_document(
                request=documentai.ProcessRequest(
                    name=self.processor_name,
                    raw_document=documentai.RawDocument(
                        content=document_content,
                        mime_type=mime_type,
                    ),
                    process_options=self._create_process_options(config),
                    skip_human_review=True,
                ),
            )

            # Convert to our standard page-based format
            pages = []
            for page in response.document.pages:
                page_dict = {
                    "content": self._get_page_full_content(response.document.text, page),
                    "paragraphs": self._get_page_paragraphs(response.document.text, page),
                    "tables": self._get_page_tables(response.document.text, page)
                }

                # If vision mode is enabled, add page image
                if self.vision_mode:
                    images_dict = self.convert_to_images(source)
                    if page.page_number - 1 in images_dict:  # Document AI uses 1-based page numbers
                        page_dict["image"] = images_dict[page.page_number - 1]

                pages.append(page_dict)

            return pages

        except Exception as e:
            raise ValueError(f"Error processing document: {str(e)}")

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

    def can_handle_vision(self, source: Union[str, BytesIO]) -> bool:
        """Check if this loader can handle the source in vision mode."""
        return self.can_handle(source)

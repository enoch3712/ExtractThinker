import mimetypes
from typing import Optional, Any, Dict, List, Union, Sequence
from io import BytesIO
from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1 as documentai
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader
import json
import os
from google.oauth2 import service_account
from cachetools.keys import hashkey
from cachetools import cachedmethod
from operator import attrgetter
import warnings

class Config:
    def __init__(
        self,
        enable_native_pdf_parsing: bool = False,
        page_range: Optional[List[int]] = None
    ):
        self.enable_native_pdf_parsing = enable_native_pdf_parsing
        self.page_range = page_range


class DocumentLoaderGoogleDocumentAI(CachedDocumentLoader):
    """Loader for documents using Google Document AI."""
    
    SUPPORTED_FORMATS = [
        # Images
        "jpeg", "jpg", "png", "bmp", "tiff", "tif", "gif", "webp",
        # Documents
        "pdf", "docx", "xlsx", "pptx", "html"
    ]

    def __init__(
        self,
        project_id: str,
        location: str,
        processor_id: str,
        credentials: str,
        processor_version: str = "rc",
    ):
        """Initialize the Google Document AI loader.
        
        Args:
            project_id: Google Cloud project ID
            location: Google Cloud location (e.g., 'us' or 'eu')
            processor_id: Document AI processor ID
            credentials: Path to service account JSON file or JSON string
            processor_version: Processor version (default: 'rc')
        """
        super().__init__()
        self.project_id = project_id
        self.location = location
        self.processor_id = processor_id
        self.processor_version = processor_version
        self.credentials = self._parse_credentials(credentials)
        self.client = self._create_client()

    def _create_client(self) -> documentai.DocumentProcessorServiceClient:
        return documentai.DocumentProcessorServiceClient(
            credentials=self.credentials,
            client_options=ClientOptions(
                api_endpoint=f"{self.location}-documentai.googleapis.com"
            )
        )

    @staticmethod
    def _parse_credentials(credentials: str) -> service_account.Credentials:
        """Parse credentials from file path or JSON string."""
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

    @staticmethod
    def _get_page_full_content(full_text: str, page: documentai.Document.Page) -> str:
        """Extract full text content from a page using token information."""
        start_index = page.tokens[0].layout.text_anchor.text_segments[0].start_index
        end_index = page.tokens[-1].layout.text_anchor.text_segments[-1].end_index
        return full_text[start_index:end_index]

    @cachedmethod(cache=attrgetter('cache'), 
                key=lambda self, source: hashkey(source if isinstance(source, str) else source.getvalue(), self.vision_mode))
    def load(self, source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        """Load and analyze a document using Google Document AI."""
        if not self.can_handle(source):
            raise ValueError(f"Cannot handle source: {source}")

        try:
            # Get document content and mime type
            if isinstance(source, str):
                with open(source, "rb") as document:
                    document_content = document.read()
                mime_type = mimetypes.guess_type(source)[0]
            else:
                document_content = source.read()
                mime_type = mimetypes.guess_type(source.name)[0] if hasattr(source, 'name') else 'application/pdf'

            # Get processor name
            name = self.client.processor_version_path(
                self.project_id, self.location, self.processor_id, self.processor_version
            )

            # Process with Document AI
            response = self.client.process_document(
                request=documentai.ProcessRequest(
                    name=name,
                    raw_document=documentai.RawDocument(
                        content=document_content,
                        mime_type=mime_type,
                    )
                )
            )

            # Convert to simplified page format
            pages = [{
                "content": self._get_page_full_content(response.document.text, page),
                "tables": self._get_page_tables(response.document.text, page),
                "forms": self._get_page_forms(page) if hasattr(page, 'form_fields') else [],
                "key_value_pairs": self._get_page_key_value_pairs(page) if hasattr(page, 'key_value_pairs') else []
            } for page in response.document.pages]

            # Add image data if in vision mode
            if self.vision_mode and self.can_handle_vision(source):
                images_dict = self.convert_to_images(source)
                for idx, page_data in enumerate(pages):
                    if idx in images_dict:
                        page_data["image"] = images_dict[idx]

            return pages

        except Exception as e:
            raise ValueError(f"Error processing document: {str(e)}")

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

    @staticmethod
    def _get_page_forms(page: documentai.Document.Page) -> List[Dict[str, str]]:
        """Extract form fields from the page."""
        return [
            {
                "name": field.field_name.text_anchor.content,
                "value": field.field_value.text_anchor.content
            }
            for field in page.form_fields
        ]

    @staticmethod
    def _get_page_key_value_pairs(page: documentai.Document.Page) -> List[Dict[str, str]]:
        """Extract key-value pairs from the page."""
        return [
            {
                "key": kv_pair.key.text_anchor.content,
                "value": kv_pair.value.text_anchor.content if kv_pair.value.text_anchor else ""
            }
            for kv_pair in page.key_value_pairs
        ]


# Create an alias with deprecation warning
class DocumentLoaderDocumentAI(DocumentLoaderGoogleDocumentAI):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "DocumentLoaderDocumentAI is deprecated and will be removed in 0.1.0"
            "Use DocumentLoaderGoogleDocumentAI instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)
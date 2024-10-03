import io
from typing import Any, Dict, List, Union

import pdfplumber

from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader
from extract_thinker.utils import get_file_extension

SUPPORTED_FORMATS = ['pdf']

class DocumentLoaderPdfPlumber(CachedDocumentLoader):
    def __init__(self, content: Any = None, cache_ttl: int = 300):
        super().__init__(content, cache_ttl)

    def load_content_from_file(self, file_path: str) -> Union[str, Dict[str, Any]]:
        try:
            if get_file_extension(file_path).lower() not in SUPPORTED_FORMATS:
                raise Exception(f"Unsupported file type: {file_path}")

            with pdfplumber.open(file_path) as pdf:
                return self.extract_data_from_pdf(pdf)
        except Exception as e:
            raise Exception(f"Error processing file: {e}") from e

    def load_content_from_stream(self, stream: io.BytesIO) -> Union[str, Dict[str, Any]]:
        try:
            with pdfplumber.open(stream) as pdf:
                return self.extract_data_from_pdf(pdf)
        except Exception as e:
            raise Exception(f"Error processing stream: {e}") from e

    def extract_data_from_pdf(self, pdf: pdfplumber.PDF) -> Dict[str, Any]:
        document_data = {
            "text": [],
            "tables": []
        }

        for page in pdf.pages:
            # Extract text
            page_text = page.extract_text()
            if page_text:
                document_data["text"].extend(page_text.split('\n'))

            # Extract tables
            tables = page.extract_tables()
            for table in tables:
                document_data["tables"].append(table)

        return document_data

    def load_content_from_file_list(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        return [self.load_content_from_file(file_path) for file_path in file_paths]

    def load_content_from_stream_list(self, streams: List[io.BytesIO]) -> List[Dict[str, Any]]:
        return [self.load_content_from_stream(stream) for stream in streams]
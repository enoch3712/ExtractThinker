import io
from typing import Any, Dict, List, Union
from pypdf import PdfReader
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader

class DocumentLoaderPyPdf(CachedDocumentLoader):
    SUPPORTED_FORMATS = ['pdf']
    
    def __init__(self, content: Any = None, cache_ttl: int = 300):
        super().__init__(content, cache_ttl)

    def load_content_from_file(self, file_path: str) -> Union[str, Dict[str, Any]]:
        reader = PdfReader(file_path)
        return self.extract_data_from_pdf(reader)

    def load_content_from_stream(self, stream: io.BytesIO) -> Union[str, Dict[str, Any]]:
        reader = PdfReader(stream)
        return self.extract_data_from_pdf(reader)

    def load_content_from_file_list(self, file_paths: List[str]) -> List[Any]:
        return [self.load_content_from_file(file_path) for file_path in file_paths]

    def load_content_from_stream_list(self, streams: List[io.BytesIO]) -> List[Any]:
        return [self.load_content_from_stream(stream) for stream in streams]

    def extract_data_from_pdf(self, reader: PdfReader) -> Union[str, Dict[str, Any]]:
        document_data = {
            "text": []
        }

        for page in reader.pages:
            # Extract text and split by newline characters
            page_text = page.extract_text()
            document_data["text"].extend(page_text.split('\n'))

        # Skip image extraction for now. TODO
        # for img_index, image in enumerate(page.images):
        #     image_data = self.extract_image_content(io.BytesIO(image["data"]))
        #     if image_data:
        #         document_data["images"].append(image_data)

        return document_data
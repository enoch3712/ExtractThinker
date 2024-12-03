from typing import List
from io import BytesIO
import docx2txt
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader

class DocumentLoaderDoc2txt(CachedDocumentLoader):
    """Loader for Microsoft Word documents."""
    
    SUPPORTED_FORMATS = ['docx', 'doc']

    def load_content_from_file(self, file_path: str) -> str:
        """Load content from a Word document file."""
        try:
            return docx2txt.process(file_path)
        except Exception as e:
            raise ValueError(f"Error loading Word document: {str(e)}")

    def load_content_from_stream(self, stream: BytesIO) -> str:
        """Load content from a Word document stream."""
        try:
            return docx2txt.process(stream)
        except Exception as e:
            raise ValueError(f"Error loading Word document from stream: {str(e)}")

    def load_content_from_stream_list(self, stream: BytesIO) -> List[str]:
        """Load content as a list from a Word document stream."""
        content = self.load_content_from_stream(stream)
        return [paragraph.strip() for paragraph in content.split('\n') if paragraph.strip()]

    def load_content_from_file_list(self, file_path: str) -> List[str]:
        """Load content as a list from a Word document file."""
        content = self.load_content_from_file(file_path)
        return [paragraph.strip() for paragraph in content.split('\n') if paragraph.strip()]

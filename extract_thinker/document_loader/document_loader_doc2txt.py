from typing import Any, Dict, List, Union
from io import BytesIO
import docx2txt

from extract_thinker.document_loader.document_loader import DocumentLoader


class DocumentLoaderDoc2txt(DocumentLoader):
    """Loader for Microsoft Word documents."""
    
    SUPPORTED_FORMATS = ['docx', 'doc']

    def load(self, source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        """
        Load content from a Word document and convert it to our standard format.
        Since Word documents don't have a clear page structure, we treat paragraphs
        as separate "pages" for consistency.

        Args:
            source: Either a file path or BytesIO stream
            
        Returns:
            List[Dict[str, Any]]: List of pages, each containing content and optional image
        """
        if not self.can_handle(source):
            raise ValueError(f"Cannot handle source: {source}")

        try:
            # Extract text content
            content = docx2txt.process(source)
            
            # Split into paragraphs and filter empty ones
            paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
            
            # Convert to our standard page-based format
            pages = []
            for paragraph in paragraphs:
                page_dict = {
                    "content": paragraph
                }
                pages.append(page_dict)

            return pages

        except Exception as e:
            raise ValueError(f"Error loading Word document: {str(e)}")

    def can_handle_vision(self, source: Union[str, BytesIO]) -> bool:
        """Word documents don't support vision mode directly."""
        return False

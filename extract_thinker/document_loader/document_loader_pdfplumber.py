import io
from typing import Any, Dict, List, Union
import pdfplumber
from operator import attrgetter
from cachetools import cachedmethod
from cachetools.keys import hashkey
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader

class DocumentLoaderPdfPlumber(CachedDocumentLoader):
    """Loader for PDFs using pdfplumber, supporting text and table extraction."""
    SUPPORTED_FORMATS = ['pdf']

    def __init__(self, content: Any = None, cache_ttl: int = 300):
        super().__init__(content, cache_ttl)

    @cachedmethod(cache=attrgetter('cache'), 
                  key=lambda self, source: hashkey(source if isinstance(source, str) else source.getvalue(), self.vision_mode))
    def load(self, source: Union[str, io.BytesIO]) -> List[Dict[str, Any]]:
        """
        Load a PDF and extract text and tables from each page.
        Returns a list of pages, each containing:
        - content: The text content of the page
        - tables: Any tables found on the page
        - image: The rendered page image (if vision_mode is True)
        
        Args:
            source: Either a file path or BytesIO stream
            
        Returns:
            List[Dict[str, Any]]: List of pages with content and optional images
        """
        if not self.can_handle(source):
            raise ValueError(f"Cannot handle source: {source}")

        try:
            # Open PDF with pdfplumber
            with pdfplumber.open(source) as pdf:
                pages = []
                for page_num, page in enumerate(pdf.pages):
                    # Extract text and tables for this page
                    page_dict = {
                        "content": page.extract_text() or "",
                        "tables": [table for table in page.extract_tables()]
                    }

                    # If vision mode is enabled, add page image
                    if self.vision_mode:
                        # Use the base class's convert_to_images method
                        images_dict = self.convert_to_images(source)
                        if page_num in images_dict:
                            page_dict["image"] = images_dict[page_num]

                    pages.append(page_dict)

                return pages

        except Exception as e:
            raise ValueError(f"Error processing PDF: {str(e)}")

    def can_handle_vision(self, source: Union[str, io.BytesIO]) -> bool:
        """Check if this loader can handle the source in vision mode."""
        return self.can_handle(source)
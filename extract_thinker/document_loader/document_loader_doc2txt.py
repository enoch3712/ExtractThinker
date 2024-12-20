from typing import Any, Dict, List, Union
from io import BytesIO
import docx2txt
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader
from cachetools import cachedmethod
from cachetools.keys import hashkey
from operator import attrgetter

class DocumentLoaderDoc2txt(CachedDocumentLoader):
    """Loader for Microsoft Word documents."""
    
    SUPPORTED_FORMATS = ['docx', 'doc']

    @cachedmethod(cache=attrgetter('cache'),
                  key=lambda self, source: hashkey(source if isinstance(source, str) else source.getvalue(), self.vision_mode))
    def load(self, source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        """
        Load content from a Word document and convert it to our standard format.
        Each page from the Word document will be treated as a separate page in the output.

        Args:
            source: Either a file path or BytesIO stream
            
        Returns:
            List[Dict[str, Any]]: List of pages, each containing content and optional image
        """
        if not self.can_handle(source):
            raise ValueError(f"Cannot handle source: {source}")

        if self.vision_mode and not self.can_handle_vision(source):
            raise ValueError(f"Cannot handle source in vision mode: {source}")

        try:
            # Extract text content
            content = docx2txt.process(source)
            
            # Split into pages using double newlines as separator
            pages_content = content.split('\n\n\n')
            
            # Convert to our standard page-based format
            pages = []
            for page_content in pages_content:
                # Skip empty pages
                if page_content.strip():
                    page_dict = {
                        "content": page_content.strip()
                    }
                    pages.append(page_dict)

            return pages

        except Exception as e:
            raise ValueError(f"Error loading Word document: {str(e)}")

    def can_handle_vision(self, source: Union[str, BytesIO]) -> bool:
        """Word documents don't support vision mode directly."""
        return False
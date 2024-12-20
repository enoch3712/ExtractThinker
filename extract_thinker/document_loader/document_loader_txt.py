from io import BytesIO
from typing import Any, Dict, List, Union
from operator import attrgetter
from cachetools import cachedmethod
from cachetools.keys import hashkey
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader
from extract_thinker.utils import get_file_extension


class DocumentLoaderTxt(CachedDocumentLoader):
    """Document loader for text files."""
    
    SUPPORTED_FORMATS = ["txt"]
    
    def __init__(self, content: Any = None, cache_ttl: int = 300):
        super().__init__(content, cache_ttl)

    @cachedmethod(cache=attrgetter('cache'), 
                  key=lambda self, source: hashkey(source if isinstance(source, str) else source.getvalue(), self.vision_mode))
    def load(self, source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        """
        Load content from a text file and convert it to our standard format.
        Since text files don't have a clear page structure, we treat paragraphs
        as separate "pages" for consistency.

        Args:
            source: Either a file path or BytesIO stream
            
        Returns:
            List[Dict[str, Any]]: List of pages, each containing content
        """
        if not self.can_handle(source):
            raise ValueError(f"Cannot handle source: {source}")
        
        # If in vision mode and can't handle vision, raise ValueError
        if self.vision_mode and not self.can_handle_vision(source):
            raise ValueError(f"Cannot handle source in vision mode: {source}")

        try:
            # Load content based on source type
            if isinstance(source, str):
                file_type = get_file_extension(source)
                if file_type.lower() not in self.SUPPORTED_FORMATS:
                    raise ValueError(f"Unsupported file type: {file_type}")
                
                with open(source, 'r', encoding='utf-8') as file:
                    content = file.read()
            else:
                # BytesIO stream
                source.seek(0)
                content = source.read().decode('utf-8')

            # Instead of splitting into paragraphs, keep everything as one content
            content = content.strip()
            
            # Return single page with all content
            return [{
                "content": content
            }]

        except Exception as e:
            raise ValueError(f"Error loading text file: {str(e)}")

    def can_handle_vision(self, source: Union[str, BytesIO]) -> bool:
        """Text files don't support vision mode."""
        return False
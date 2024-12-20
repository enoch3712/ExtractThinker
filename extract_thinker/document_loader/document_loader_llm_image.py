from typing import Any, Dict, List, Union
from io import BytesIO
from operator import attrgetter
from cachetools import cachedmethod
from cachetools.keys import hashkey
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader


class DocumentLoaderLLMImage(CachedDocumentLoader):
    """
    Document loader that handles images and PDFs, converting them to a format suitable for vision LLMs.
    This loader is used as a fallback when no other loader is available and vision mode is required.
    """
    SUPPORTED_FORMATS = ['pdf', 'jpg', 'jpeg', 'png', 'tiff', 'bmp']
    
    def __init__(self, content=None, cache_ttl=300, llm=None):
        super().__init__(content, cache_ttl)
        self.llm = llm
        self.vision_mode = True  # Always in vision mode since this is for image processing

    @cachedmethod(cache=attrgetter('cache'), 
                  key=lambda self, source: hashkey(source if isinstance(source, str) else source.getvalue(), self.vision_mode))
    def load(self, source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        """
        Load the source and convert it to a list of pages with images.
        Each page will be a dictionary with:
        - 'content': Empty string (since this loader doesn't extract text)
        - 'image': The image bytes for that page
        
        Args:
            source: Either a file path or a BytesIO stream
            
        Returns:
            List[Dict[str, Any]]: List of pages, each with 'content' and 'image' keys
        """
        if not self.can_handle(source):
            raise ValueError(f"Cannot handle source: {source}")

        try:
            # Convert source to images using the base class's convert_to_images method
            images_dict = self.convert_to_images(source)
            
            # Convert to our standard page-based format
            pages = []
            for page_idx, image_bytes in images_dict.items():
                pages.append({
                    "content": "",  # No text content since this is image-only
                    "image": image_bytes
                })
            
            return pages
            
        except Exception as e:
            raise ValueError(f"Failed to load image content: {str(e)}")

    def can_handle_vision(self, source: Union[str, BytesIO]) -> bool:
        """
        Check if this loader can handle the source in vision mode.
        This loader is specifically for vision/image processing.
        """
        return self.can_handle(source)
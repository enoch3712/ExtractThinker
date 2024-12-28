from io import BytesIO
from typing import Any, Dict, List, Union
from operator import attrgetter
from cachetools import cachedmethod
from cachetools.keys import hashkey
import magic

from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader
from extract_thinker.utils import MIME_TYPE_MAPPING

try:
    from markitdown import MarkItDown
except ImportError:
    raise ImportError("MarkItDown library is not installed. Please install it with 'pip install markitdown'.")

class DocumentLoaderMarkItDown(CachedDocumentLoader):
    """
    Document loader that uses MarkItDown to extract content from various file formats.
    Supports text extraction and optional image/page rendering in vision mode.
    """
    
    SUPPORTED_FORMATS = [
        "pdf", "doc", "docx", "ppt", "pptx", "xls", "xlsx", 
        "csv", "tsv", "txt", "html", "xml", "json", "zip",
        "jpg", "jpeg", "png", "bmp", "gif", "wav", "mp3", "m4a"
    ]
    
    def __init__(self, content: Any = None, cache_ttl: int = 300, llm_client=None, llm_model=None):
        super().__init__(content, cache_ttl)
        self.markitdown = MarkItDown(llm_client=llm_client, llm_model=llm_model)

    @cachedmethod(cache=attrgetter('cache'), 
                  key=lambda self, source: hashkey(source if isinstance(source, str) else source.getvalue(), self.vision_mode))
    def load(self, source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        """
        Load and process content using MarkItDown.
        Returns a list of pages, each containing:
        - content: The text content
        - image: The page/image bytes if vision_mode is True
        
        Args:
            source: Either a file path or BytesIO stream
            
        Returns:
            List[Dict[str, Any]]: List of pages with content and optional images
        """
        if not self.can_handle(source):
            raise ValueError(f"Cannot handle source: {source}")

        if self.vision_mode and not self.can_handle_vision(source):
            raise ValueError(f"Cannot handle source in vision mode: {source}")

        try:
            # Extract text content using MarkItDown
            if isinstance(source, str):
                result = self.markitdown.convert(source)
            else:
                # For BytesIO, we need to determine the file type
                source.seek(0)
                mime = magic.from_buffer(source.getvalue(), mime=True)
                ext = next((ext for ext, mime_types in MIME_TYPE_MAPPING.items() 
                          if mime in (mime_types if isinstance(mime_types, list) else [mime_types])), 'txt')
                result = self.markitdown.convert_stream(source, file_extension=f".{ext}")
                source.seek(0)

            text_content = result.text_content

            # Split into pages if supported
            pages = []
            if self.can_handle_paginate(source):
                raw_pages = text_content.split("\f")
                for page_text in raw_pages:
                    if page_text.strip():
                        pages.append({"content": page_text.strip()})
            else:
                pages = [{"content": text_content.strip()}]

            # Add images in vision mode
            if self.vision_mode:
                images_dict = self.convert_to_images(source)
                for idx, page_dict in enumerate(pages):
                    if idx in images_dict:
                        page_dict["image"] = images_dict[idx]

            return pages

        except Exception as e:
            raise ValueError(f"Error processing document with MarkItDown: {str(e)}")
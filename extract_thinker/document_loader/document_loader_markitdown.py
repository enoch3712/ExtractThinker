import io
from io import BytesIO
from typing import Any, Dict, List, Union, Optional
from operator import attrgetter
from cachetools import cachedmethod
from cachetools.keys import hashkey
import magic
from dataclasses import dataclass, field

from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader
from extract_thinker.utils import MIME_TYPE_MAPPING


@dataclass
class MarkItDownConfig:
    """Configuration for MarkItDown document loader.
    
    Args:
        content: Initial content (optional)
        cache_ttl: Cache time-to-live in seconds (default: 300)
        llm_client: LLM client for enhanced text processing (optional)
        llm_model: LLM model name to use (optional)
        mime_type_detection: Whether to use magic for MIME type detection (default: True)
        default_extension: Default file extension when type cannot be determined (default: 'txt')
        page_separator: Character used to separate pages (default: form feed '\\f')
        preserve_whitespace: Whether to preserve whitespace in text (default: False)
    """
    # Optional parameters
    content: Optional[Any] = None
    cache_ttl: int = 300
    llm_client: Optional[Any] = None
    llm_model: Optional[str] = None
    mime_type_detection: bool = True
    default_extension: str = 'txt'
    page_separator: str = '\f'
    preserve_whitespace: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not isinstance(self.mime_type_detection, bool):
            raise ValueError("mime_type_detection must be a boolean")
        
        if not self.default_extension:
            raise ValueError("default_extension cannot be empty")
        
        if not self.page_separator:
            raise ValueError("page_separator cannot be empty")


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
    
    def __init__(
        self,
        content_or_config: Union[Any, MarkItDownConfig] = None,
        cache_ttl: int = 300,
        llm_client: Optional[Any] = None,
        llm_model: Optional[str] = None,
        mime_type_detection: bool = True,
        default_extension: str = 'txt',
        page_separator: str = '\f',
        preserve_whitespace: bool = False
    ):
        """Initialize loader.
        
        Args:
            content_or_config: Either a MarkItDownConfig object or initial content
            cache_ttl: Cache time-to-live in seconds (only used if content_or_config is not MarkItDownConfig)
            llm_client: LLM client for enhanced text processing (only used if content_or_config is not MarkItDownConfig)
            llm_model: LLM model name to use (only used if content_or_config is not MarkItDownConfig)
            mime_type_detection: Whether to use magic for MIME type detection (only used if content_or_config is not MarkItDownConfig)
            default_extension: Default file extension when type cannot be determined (only used if content_or_config is not MarkItDownConfig)
            page_separator: Character used to separate pages (only used if content_or_config is not MarkItDownConfig)
            preserve_whitespace: Whether to preserve whitespace in text (only used if content_or_config is not MarkItDownConfig)
        """
        # Check dependencies before initializing
        self._check_dependencies()
        
        # Handle both config-based and old-style initialization
        if isinstance(content_or_config, MarkItDownConfig):
            self.config = content_or_config
        else:
            # Create config from individual parameters
            self.config = MarkItDownConfig(
                content=content_or_config,
                cache_ttl=cache_ttl,
                llm_client=llm_client,
                llm_model=llm_model,
                mime_type_detection=mime_type_detection,
                default_extension=default_extension,
                page_separator=page_separator,
                preserve_whitespace=preserve_whitespace
            )
        
        super().__init__(self.config.content, self.config.cache_ttl)
        self.markitdown = self._get_markitdown()(
            llm_client=self.config.llm_client,
            llm_model=self.config.llm_model
        )

    @staticmethod
    def _check_dependencies():
        """Check if required dependencies are installed."""
        try:
            import markitdown
        except ImportError:
            raise ImportError(
                "Could not import markitdown package. "
                "Please install it with `pip install markitdown`."
            )

    def _get_markitdown(self):
        """Lazy load MarkItDown."""
        try:
            from markitdown import MarkItDown
            return MarkItDown
        except ImportError:
            raise ImportError(
                "Could not import markitdown python package. "
                "Please install it with `pip install markitdown`."
            )

    def _process_text(self, text: str) -> str:
        """Process text according to configuration."""
        if not self.config.preserve_whitespace:
            text = text.strip()
        return text

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
                if self.config.mime_type_detection:
                    mime = magic.from_buffer(source.getvalue(), mime=True)
                    ext = next((ext for ext, mime_types in MIME_TYPE_MAPPING.items() 
                              if mime in (mime_types if isinstance(mime_types, list) else [mime_types])), 
                             self.config.default_extension)
                else:
                    ext = self.config.default_extension
                result = self.markitdown.convert_stream(source, file_extension=f".{ext}")
                source.seek(0)

            text_content = result.text_content

            # Split into pages if supported
            pages = []
            if self.can_handle_paginate(source):
                raw_pages = text_content.split(self.config.page_separator)
                for page_text in raw_pages:
                    processed_text = self._process_text(page_text)
                    if processed_text or self.config.preserve_whitespace:
                        pages.append({"content": processed_text})
            else:
                processed_text = self._process_text(text_content)
                pages = [{"content": processed_text}]

            # Add images in vision mode
            if self.vision_mode:
                images_dict = self.convert_to_images(source)
                for idx, page_dict in enumerate(pages):
                    if idx in images_dict:
                        page_dict["image"] = images_dict[idx]

            return pages

        except Exception as e:
            raise ValueError(f"Error processing document with MarkItDown: {str(e)}")
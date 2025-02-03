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
    """
    Configuration for MarkItDown document loader.

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
    Produces a list of pages, each with:
      - "content": text from that page
      - "image": optional page/image bytes if vision_mode is True
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
        """
        Initialize the loader.

        Args:
            content_or_config: Either a MarkItDownConfig object or the initial content
            cache_ttl: Cache time-to-live in seconds (only used if content_or_config is not MarkItDownConfig)
            llm_client: LLM client (only used if content_or_config is not MarkItDownConfig)
            llm_model: LLM model name (only used if content_or_config is not MarkItDownConfig)
            mime_type_detection: Whether to use magic for MIME type detection
            default_extension: Default extension if MIME type detection fails
            page_separator: Character used to separate pages
            preserve_whitespace: Whether to preserve whitespace
        """
        self._check_dependencies()

        # Handle config object vs. old-style params
        if isinstance(content_or_config, MarkItDownConfig):
            self.config = content_or_config
        else:
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

        # MarkItDown object
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
                "Could not import the 'markitdown' package. "
                "Please install it with `pip install markitdown`."
            )

    def _get_markitdown(self):
        """Lazy-import MarkItDown class."""
        from markitdown import MarkItDown
        return MarkItDown

    def _process_text(self, text: str) -> str:
        """Apply any additional text processing (e.g., strip whitespace)."""
        return text if self.config.preserve_whitespace else text.strip()

    def _is_url(self, source: str) -> bool:
        """Check if the source is a URL."""
        try:
            from urllib.parse import urlparse
            result = urlparse(source)
            return all([result.scheme, result.netloc])
        except:
            return False

    @cachedmethod(cache=attrgetter('cache'), 
                  key=lambda self, source: hashkey(
                      source if isinstance(source, str) else source.getvalue(), 
                      self.vision_mode
                  ))
    def load(self, source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        """
        Load and process the source with MarkItDown, returning a list of pages.

        Args:
            source: A file path, BytesIO stream, or URL

        Returns:
            A list of dictionaries where each dict is one "page" of text.
            - "content": The text content (str)
            - "image": Optional bytes if vision mode is enabled (key only present if vision_mode is True)
        """
        if not self.can_handle(source):
            raise ValueError(f"Cannot handle source: {source}")

        # Basic check for vision mode
        if self.vision_mode and not self.can_handle_vision(source):
            raise ValueError(f"Cannot handle source in vision mode: {source}")

        try:
            # Convert the file or stream with MarkItDown
            if isinstance(source, str):
                # File path
                result = self.markitdown.convert(source)
            else:
                # BytesIO
                source.seek(0)
                if self.config.mime_type_detection:
                    mime = magic.from_buffer(source.getvalue(), mime=True)
                    # Attempt to deduce extension from MIME type
                    ext = next(
                        (
                            e
                            for e, mime_list in MIME_TYPE_MAPPING.items()
                            if mime in (mime_list if isinstance(mime_list, list) else [mime_list])
                        ),
                        self.config.default_extension
                    )
                else:
                    ext = self.config.default_extension
                result = self.markitdown.convert_stream(source, file_extension=f".{ext}")
                source.seek(0)

            # Full text from MarkItDown
            text_content = result.text_content
            if not text_content:
                text_content = ""

            # Split text content into pages (based on config.page_separator)
            raw_pages = text_content.split(self.config.page_separator)

            pages = []
            for page_text in raw_pages:
                processed = self._process_text(page_text)
                # Always include the page if preserve_whitespace is True, 
                # or if there's any non-empty text.
                if processed or self.config.preserve_whitespace:
                    pages.append({"content": processed})

            # In vision mode, attach rendered images if applicable
            if self.vision_mode:
                images_dict = self.convert_to_images(source)
                # Match up page images by index
                for idx, page_dict in enumerate(pages):
                    if idx in images_dict:
                        page_dict["image"] = images_dict[idx]

            return pages

        except Exception as e:
            raise ValueError(f"Error processing document with MarkItDown: {str(e)}")

    def can_handle(self, source: Union[str, BytesIO]) -> bool:
        """
        Checks if the loader can handle the given source.
        
        Args:
            source: Either a file path (str), a BytesIO stream, or a URL
            
        Returns:
            bool: True if the loader can handle the source, False otherwise
        """
        try:
            if isinstance(source, str):
                if self._is_url(source):
                    return True
                return self._can_handle_file_path(source)
            elif isinstance(source, BytesIO):
                return self._can_handle_stream(source)
            return False
        except Exception:
            return False
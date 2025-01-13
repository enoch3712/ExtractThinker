from typing import Any, Dict, List, Union, Optional
from io import BytesIO
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader
from cachetools import cachedmethod
from cachetools.keys import hashkey
from operator import attrgetter
from dataclasses import dataclass
import os


@dataclass
class Doc2txtConfig:
    """Configuration for Doc2txt loader.
    
    Args:
        content: Initial content (optional)
        cache_ttl: Cache time-to-live in seconds (default: 300)
        page_separator: String to use for separating pages (default: '\n\n\n')
        preserve_whitespace: Whether to preserve whitespace in extracted text (default: False)
        extract_images: Whether to extract embedded images (not supported, always False)
    """
    content: Optional[Any] = None
    cache_ttl: int = 300
    page_separator: str = '\n\n\n'
    preserve_whitespace: bool = False
    extract_images: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.cache_ttl <= 0:
            raise ValueError("cache_ttl must be positive")
        
        if not isinstance(self.page_separator, str):
            raise ValueError("page_separator must be a string")
        
        if self.extract_images:
            raise ValueError("Image extraction is not supported for Word documents")


class DocumentLoaderDoc2txt(CachedDocumentLoader):
    """Loader for Microsoft Word documents."""
    
    SUPPORTED_FORMATS = ['docx', 'doc']

    def __init__(
        self,
        content_or_config: Union[Any, Doc2txtConfig] = None,
        cache_ttl: int = 300,
        page_separator: str = '\n\n\n',
        preserve_whitespace: bool = False,
        extract_images: bool = False
    ):
        """Initialize loader.
        
        Args:
            content_or_config: Either a Doc2txtConfig object or initial content
            cache_ttl: Cache time-to-live in seconds (only used if content_or_config is not Doc2txtConfig)
            page_separator: String to use for separating pages (only used if content_or_config is not Doc2txtConfig)
            preserve_whitespace: Whether to preserve whitespace (only used if content_or_config is not Doc2txtConfig)
            extract_images: Whether to extract images (not supported, only used if content_or_config is not Doc2txtConfig)
        """
        # Check required dependencies
        self._check_dependencies()

        # Handle both config-based and old-style initialization
        if isinstance(content_or_config, Doc2txtConfig):
            self.config = content_or_config
        else:
            # Create config from individual parameters
            self.config = Doc2txtConfig(
                content=content_or_config,
                cache_ttl=cache_ttl,
                page_separator=page_separator,
                preserve_whitespace=preserve_whitespace,
                extract_images=extract_images
            )
        
        super().__init__(self.config.content, self.config.cache_ttl)
        self.vision_mode = False  # Word documents don't support vision mode

    @staticmethod
    def _check_dependencies():
        """Check if required dependencies are installed."""
        try:
            import docx2txt
        except ImportError:
            raise ImportError(
                "Could not import docx2txt python package. "
                "Please install it with `pip install docx2txt`."
            )

    def _get_docx2txt(self):
        """Lazy load docx2txt."""
        try:
            import docx2txt
            return docx2txt
        except ImportError:
            raise ImportError(
                "Could not import docx2txt python package. "
                "Please install it with `pip install docx2txt`."
            )

    @cachedmethod(cache=attrgetter('cache'),
                  key=lambda self, source: hashkey(source if isinstance(source, str) else source.getvalue(), self.vision_mode))
    def load(self, source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        """
        Load content from a Word document and convert it to our standard format.
        Each page from the Word document will be treated as a separate page in the output.

        Args:
            source: Either a file path or BytesIO stream
            
        Returns:
            List[Dict[str, Any]]: List of pages, each containing content
        """
        if not self.can_handle(source):
            raise ValueError(f"Cannot handle source: {source}")

        docx2txt = self._get_docx2txt()

        try:
            # Extract text content
            content = docx2txt.process(source)
            
            # Split into pages using configured separator
            pages_content = content.split(self.config.page_separator)
            
            # Convert to our standard page-based format
            pages = []
            for page_content in pages_content:
                # Skip empty pages unless preserve_whitespace is True
                if page_content.strip() or self.config.preserve_whitespace:
                    page_dict = {
                        "content": page_content if self.config.preserve_whitespace else page_content.strip()
                    }
                    pages.append(page_dict)

            return pages

        except Exception as e:
            raise ValueError(f"Error loading Word document: {str(e)}")

    def can_handle(self, source: Union[str, BytesIO]) -> bool:
        """Check if this loader can handle the source."""
        if isinstance(source, str):
            # Check if it's a valid file path
            if not os.path.exists(source):
                return False
            # Check extension
            ext = os.path.splitext(source)[1].lower().lstrip('.')
            return ext in self.SUPPORTED_FORMATS
        elif isinstance(source, BytesIO):
            # For BytesIO, we can only check if it's not empty
            try:
                source.seek(0, 2)  # Seek to end
                size = source.tell()
                source.seek(0)  # Reset position
                return size > 0
            except Exception:
                return False
        return False

    def can_handle_vision(self, source: Union[str, BytesIO]) -> bool:
        """Word documents don't support vision mode directly."""
        return False

    def set_vision_mode(self, enabled: bool = True):
        """Vision mode is not supported for Word documents."""
        if enabled:
            raise ValueError("Vision mode is not supported for Word documents")
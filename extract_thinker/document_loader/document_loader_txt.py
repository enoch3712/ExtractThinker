import io
from io import BytesIO
from typing import Any, Dict, List, Union, Optional
from operator import attrgetter
from cachetools import cachedmethod
from cachetools.keys import hashkey
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader
from extract_thinker.utils import get_file_extension
from dataclasses import dataclass


@dataclass
class TxtConfig:
    """Configuration for TXT loader.
    
    Args:
        content: Initial content (optional)
        cache_ttl: Cache time-to-live in seconds (default: 300)
        encoding: Text encoding to use (default: 'utf-8')
        preserve_whitespace: Whether to preserve whitespace in text (default: False)
        split_paragraphs: Whether to split text into paragraphs (default: False)
    """
    content: Optional[Any] = None
    cache_ttl: int = 300
    encoding: str = 'utf-8'
    preserve_whitespace: bool = False
    split_paragraphs: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.cache_ttl <= 0:
            raise ValueError("cache_ttl must be positive")
        
        if not isinstance(self.encoding, str):
            raise ValueError("encoding must be a string")


class DocumentLoaderTxt(CachedDocumentLoader):
    """Document loader for text files."""
    
    SUPPORTED_FORMATS = ["txt"]
    
    def __init__(
        self,
        content_or_config: Union[Any, TxtConfig] = None,
        cache_ttl: int = 300,
        encoding: str = 'utf-8',
        preserve_whitespace: bool = False,
        split_paragraphs: bool = False
    ):
        """Initialize loader.
        
        Args:
            content_or_config: Either a TxtConfig object or initial content
            cache_ttl: Cache time-to-live in seconds (only used if content_or_config is not TxtConfig)
            encoding: Text encoding to use (only used if content_or_config is not TxtConfig)
            preserve_whitespace: Whether to preserve whitespace (only used if content_or_config is not TxtConfig)
            split_paragraphs: Whether to split text into paragraphs (only used if content_or_config is not TxtConfig)
        """
        # Handle both config-based and old-style initialization
        if isinstance(content_or_config, TxtConfig):
            self.config = content_or_config
        else:
            # Create config from individual parameters
            self.config = TxtConfig(
                content=content_or_config,
                cache_ttl=cache_ttl,
                encoding=encoding,
                preserve_whitespace=preserve_whitespace,
                split_paragraphs=split_paragraphs
            )
        
        super().__init__(self.config.content, self.config.cache_ttl)
        self.vision_mode = False  # Text files don't support vision mode

    @cachedmethod(cache=attrgetter('cache'), 
                  key=lambda self, source: hashkey(source if isinstance(source, str) else source.getvalue(), self.vision_mode))
    def load(self, source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        """
        Load content from a text file and convert it to our standard format.
        Since text files don't have a clear page structure, we treat paragraphs
        as separate "pages" for consistency if split_paragraphs is True.

        Args:
            source: Either a file path or BytesIO stream
            
        Returns:
            List[Dict[str, Any]]: List of pages, each containing content
        """
        if not self.can_handle(source):
            raise ValueError(f"Cannot handle source: {source}")

        try:
            # Load content based on source type
            if isinstance(source, str):
                file_type = get_file_extension(source)
                if file_type.lower() not in self.SUPPORTED_FORMATS:
                    raise ValueError(f"Unsupported file type: {file_type}")
                
                with open(source, 'r', encoding=self.config.encoding) as file:
                    content = file.read()
            else:
                # BytesIO stream
                source.seek(0)
                content = source.read().decode(self.config.encoding)

            if not self.config.preserve_whitespace:
                content = content.strip()

            # Split into pages if configured
            if self.config.split_paragraphs:
                # Split on double newlines to separate paragraphs
                pages_content = [p for p in content.split('\n\n') if p.strip() or self.config.preserve_whitespace]
            else:
                # Keep as single page
                pages_content = [content]
            
            # Convert to our standard page-based format
            return [{"content": page} for page in pages_content]

        except Exception as e:
            raise ValueError(f"Error loading text file: {str(e)}")

    def can_handle_vision(self, source: Union[str, BytesIO]) -> bool:
        """Text files don't support vision mode."""
        return False

    def set_vision_mode(self, enabled: bool = True):
        """Vision mode is not supported for text files."""
        if enabled:
            raise ValueError("Vision mode is not supported for text files")
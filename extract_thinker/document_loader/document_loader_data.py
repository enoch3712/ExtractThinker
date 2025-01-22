from typing import Any, Dict, List, Union, Optional, IO
from cachetools import cachedmethod
from cachetools.keys import hashkey
from operator import attrgetter
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader
from dataclasses import dataclass

@dataclass
class DataLoaderConfig:
    """Configuration for Data loader.
    
    Args:
        content: Initial content (optional)
        cache_ttl: Cache time-to-live in seconds (default: 300)
        supports_vision: Whether this loader supports vision mode (default: True)
    """
    content: Optional[Any] = None
    cache_ttl: int = 300
    supports_vision: bool = True

class DocumentLoaderData(CachedDocumentLoader):
    """
    Document loader that handles pre-processed data with caching support.
    Expects data in standard format:
    [
      {
        "content": "...some text...",
        "image": None or [] or bytes
      }
    ]
    """
    
    def __init__(self, 
                 content: Optional[Any] = None,
                 cache_ttl: int = 300,
                 supports_vision: bool = True):
        """Initialize loader with optional content and cache settings."""
        self.config = DataLoaderConfig(
            content=content,
            cache_ttl=cache_ttl,
            supports_vision=supports_vision
        )
        super().__init__(self.config.content, self.config.cache_ttl)
        self._supports_vision = self.config.supports_vision

    def can_handle(self, source: Any) -> bool:
        """Check if we can handle this source type."""
        if isinstance(source, str):
            return True
        if hasattr(source, "read"):
            return True
        if isinstance(source, list) and all(isinstance(item, dict) for item in source):
            return True
        return False

    @cachedmethod(cache=attrgetter('cache'), 
                  key=lambda self, source: hashkey(
                      source if isinstance(source, str)
                      else source.getvalue() if hasattr(source, 'getvalue')
                      else str(source),
                      self.vision_mode))
    def load(self, source: Union[str, IO, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Load and process content with caching support.
        Returns a list of pages in standard format.
        
        Args:
            source: String, IO stream, or pre-formatted list of dicts
            
        Returns:
            List[Dict[str, Any]]: List of pages with content and image
        """
        if not self.can_handle(source):
            raise ValueError("Can only handle str, readable streams, or list of dicts")

        try:
            if isinstance(source, list):
                return self._validate_and_format_list(source)
            elif isinstance(source, str):
                return self._load_from_string(source)
            elif hasattr(source, "read"):
                return self._load_from_stream(source)

        except Exception as e:
            raise ValueError(f"Error processing content: {str(e)}")

    def _validate_and_format_list(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and format a list of dictionaries."""
        formatted_pages = []
        for item in data:
            if not isinstance(item, dict):
                raise ValueError(
                    "Invalid format. Expected a list of dictionaries in format: "
                    "[{'content': 'your text here', 'image': None or [] or bytes}, ...]. "
                    f"Got item of type {type(item).__name__} instead of dict"
                )
            if "content" not in item:
                raise ValueError(
                    "Invalid format. Each dictionary must have a 'content' field. "
                    "Expected format: [{'content': 'your text here', 'image': None or [] or bytes}, ...]. "
                    f"Got keys: {list(item.keys())}"
                )
            
            # Preserve the original image value if present, otherwise use vision mode default
            image_value = item.get("image", [] if self.vision_mode else None)
            page = {
                "content": item["content"],
                "image": image_value
            }
            formatted_pages.append(page)
        return formatted_pages

    def _load_from_string(self, text: str) -> List[Dict[str, Any]]:
        """Process string input."""
        try:
            # Try to read as file first
            with open(text, "r", encoding="utf-8") as f:
                content = f.read()
        except (FileNotFoundError, IOError):
            # If not a file, treat as raw text
            content = text

        return [{
            "content": content,
            "image": [] if self.vision_mode else None
        }]

    def _load_from_stream(self, stream: IO) -> List[Dict[str, Any]]:
        """Process stream input."""
        try:
            content = stream.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')

            return [{
                "content": content,
                "image": [] if self.vision_mode else None
            }]
        except Exception as e:
            raise ValueError(f"Failed to read from stream: {str(e)}")

    def can_handle_vision(self, source: Union[str, IO]) -> bool:
        """Check if this loader can handle the source in vision mode."""
        return self._supports_vision and self.can_handle(source) 
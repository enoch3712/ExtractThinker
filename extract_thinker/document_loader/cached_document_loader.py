from io import BytesIO
from typing import Any, Union
from cachetools import TTLCache
from extract_thinker.document_loader.document_loader import DocumentLoader


class CachedDocumentLoader(DocumentLoader):
    def __init__(self, content: Any = None, cache_ttl: int = 300):
        super().__init__(content)
        self.cache = TTLCache(maxsize=100, ttl=cache_ttl)

    def load(self, source: Union[str, BytesIO]) -> Any:
        """
        Load content from source with caching support.
        
        Args:
            source: Either a file path (str) or a BytesIO stream
            
        Returns:
            The loaded content
        """
        # Use the source and vision_mode state as the cache key
        if isinstance(source, str):
            cache_key = (source, self.vision_mode)
        else:
            # For BytesIO, use the content and vision_mode state as the cache key
            cache_key = (source.getvalue(), self.vision_mode)

        if cache_key in self.cache:
            return self.cache[cache_key]

        result = super().load(source)
        self.cache[cache_key] = result
        return result

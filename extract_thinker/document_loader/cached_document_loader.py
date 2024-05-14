

from io import BytesIO
from typing import Any, Union

from cachetools import TTLCache
from extract_thinker.document_loader.document_loader import DocumentLoader


class CachedDocumentLoader(DocumentLoader):
    def __init__(self, content: Any = None, cache_ttl: int = 300):
        super().__init__(content)
        self.cache = TTLCache(maxsize=100, ttl=cache_ttl)

    def cached_load_content_from_file(self, file_path: str) -> Union[str, object]:
        return self.load_content_from_file(file_path)

    def cached_load_content_from_stream(self, stream: BytesIO) -> Union[str, object]:
        return self.load_content_from_stream(stream)

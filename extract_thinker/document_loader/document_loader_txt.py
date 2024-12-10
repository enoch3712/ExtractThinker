from io import BytesIO
from operator import attrgetter
from typing import Any, List, Union

from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader
from extract_thinker.utils import get_file_extension
from cachetools import cachedmethod
from cachetools.keys import hashkey

class DocumentLoaderTxt(CachedDocumentLoader):
    """Document loader for text files."""
    
    SUPPORTED_FORMATS = ["txt"]
    
    def __init__(self, content: Any = None, cache_ttl: int = 300):
        super().__init__(content, cache_ttl)

    @cachedmethod(cache=attrgetter('cache'), key=lambda self, file_path: hashkey(file_path))
    def load_content_from_file(self, file_path: str) -> Union[str, object]:
        """Load content from a text file."""
        try:
            file_type = get_file_extension(file_path)
            if file_type.lower() not in self.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            with open(file_path, 'r', encoding='utf-8') as file:
                self.content = file.read()
                return self.content
        except Exception as e:
            raise Exception(f"Error processing file: {e}") from e

    @cachedmethod(cache=attrgetter('cache'), key=lambda self, stream: hashkey(id(stream)))
    def load_content_from_stream(self, stream: BytesIO) -> Union[str, object]:
        """Load content from a BytesIO stream."""
        try:
            # Decode bytes to string
            stream.seek(0)
            self.content = stream.read().decode('utf-8')
            return self.content
        except Exception as e:
            raise Exception(f"Error processing stream: {e}") from e

    @cachedmethod(cache=attrgetter('cache'), key=lambda self, stream: hashkey(id(stream)))
    def load_content_from_stream_list(self, stream: BytesIO) -> List[Any]:
        """Load content from a stream and split into a list of chunks."""
        try:
            content = self.load_content_from_stream(stream)
            # Split content into chunks (e.g., by paragraphs or lines)
            chunks = content.split('\n\n')  # Split by double newline (paragraphs)
            return [{"content": chunk} for chunk in chunks if chunk.strip()]
        except Exception as e:
            raise Exception(f"Error processing stream list: {e}") from e

    @cachedmethod(cache=attrgetter('cache'), key=lambda self, input: hashkey(id(input)))
    def load_content_from_file_list(self, input: Union[str, List[str]]) -> List[Any]:
        """Load content from a file and split into a list of chunks."""
        try:
            if isinstance(input, str):
                content = self.load_content_from_file(input)
            elif isinstance(input, list):
                # If input is a list of files, concatenate their contents
                content = ""
                for file_path in input:
                    content += self.load_content_from_file(file_path) + "\n\n"
            else:
                raise ValueError(f"Unsupported input type: {type(input)}")

            # Split content into chunks (e.g., by paragraphs or lines)
            chunks = content.split('\n\n')  # Split by double newline (paragraphs)
            return [{"content": chunk} for chunk in chunks if chunk.strip()]
        except Exception as e:
            raise Exception(f"Error processing file list: {e}") from e

    def can_handle_vision(self, source: Union[str, BytesIO]) -> bool:
        """Text files don't support vision mode."""
        return False 
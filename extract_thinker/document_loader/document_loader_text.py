from io import BytesIO
from typing import List

from extract_thinker.document_loader.document_loader import DocumentLoader


class DocumentLoaderText(DocumentLoader):
    def __init__(self, content: str = None, cache_ttl: int = 300):
        super().__init__(content, cache_ttl)

    def load_content_from_file(self, file_path: str) -> str:
        with open(file_path, 'r') as file:
            self.content = file.read()
        return self.content

    def load_content_from_stream(self, stream: BytesIO) -> str:
        self.content = stream.getvalue().decode()
        return self.content

    def load_content_from_stream_list(self, streams: List[BytesIO]) -> List[str]:
        return [self.load_content_from_stream(stream) for stream in streams]

    def load_content_from_file_list(self, file_paths: List[str]) -> List[str]:
        return [self.load_content_from_file(file_path) for file_path in file_paths]

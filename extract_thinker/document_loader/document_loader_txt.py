from io import BytesIO
from typing import Any, Dict, List, Union

from extract_thinker.document_loader.document_loader import DocumentLoader
from extract_thinker.utils import get_file_extension


class DocumentLoaderTxt(DocumentLoader):
    """Document loader for text files."""
    
    SUPPORTED_FORMATS = ["txt"]
    
    def __init__(self, content: Any = None, cache_ttl: int = 300):
        super().__init__(content, cache_ttl)

    def load(self, source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        """
        Load content from a text file and convert it to our standard format.
        Since text files don't have a clear page structure, we treat paragraphs
        as separate "pages" for consistency.

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
                
                with open(source, 'r', encoding='utf-8') as file:
                    content = file.read()
            else:
                # BytesIO stream
                source.seek(0)
                content = source.read().decode('utf-8')

            # Split into paragraphs and filter empty ones
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            # Convert to our standard page-based format
            pages = []
            for paragraph in paragraphs:
                page_dict = {
                    "content": paragraph
                }
                pages.append(page_dict)

            return pages

        except Exception as e:
            raise ValueError(f"Error loading text file: {str(e)}")

    def can_handle_vision(self, source: Union[str, BytesIO]) -> bool:
        """Text files don't support vision mode."""
        return False 

    def load_content_from_file(self, file_path: str) -> Union[str, object]:
        """Legacy method for backward compatibility."""
        pages = self.load(file_path)
        return "\n\n".join(page["content"] for page in pages)

    def load_content_from_stream(self, stream: BytesIO) -> Union[str, object]:
        """Legacy method for backward compatibility."""
        pages = self.load(stream)
        return "\n\n".join(page["content"] for page in pages)

    def load_content_list(self, source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        """Legacy method for backward compatibility."""
        return self.load(source)

    def load_content_from_file_list(self, input: Union[str, List[str]]) -> List[Any]:
        """Legacy method for backward compatibility."""
        if isinstance(input, list):
            all_pages = []
            for file_path in input:
                pages = self.load(file_path)
                all_pages.extend(pages)
            return all_pages
        return self.load(input)

    def load_content_from_stream_list(self, stream: BytesIO) -> List[Any]:
        """Legacy method for backward compatibility."""
        return self.load(stream)
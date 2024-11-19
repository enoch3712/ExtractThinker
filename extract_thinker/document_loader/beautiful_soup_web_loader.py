from typing import List, Union, Dict, Any
from io import BytesIO
from urllib.parse import urlparse
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader
from extract_thinker.utils import get_file_extension, num_tokens_from_string

class DocumentLoaderBeautifulSoup(CachedDocumentLoader):
    """Loader that uses BeautifulSoup4 to load HTML content."""
    
    SUPPORTED_FORMATS = ['html', 'htm']
    
    def __init__(self, header_handling: str = "skip", content: Any = None, cache_ttl: int = 300):
        """Initialize loader.
        
        Args:
            header_handling: How to handle headers - "skip", "summarize", or "include"
            content: Initial content
            cache_ttl: Cache time-to-live in seconds
        """
        # Check required dependencies
        self._check_dependencies()
        super().__init__(content, cache_ttl)
        self.header_handling = header_handling

    @staticmethod
    def _check_dependencies():
        """Check if required dependencies are installed."""
        try:
            import bs4
            import requests
        except ImportError:
            raise ImportError(
                "Could not import bs4 or requests python package. "
                "Please install it with `pip install beautifulsoup4 requests`."
            )

    def _get_bs4(self):
        """Lazy load BeautifulSoup."""
        try:
            from bs4 import BeautifulSoup
            return BeautifulSoup
        except ImportError:
            raise ImportError(
                "Could not import bs4 python package. "
                "Please install it with `pip install beautifulsoup4`."
            )

    def _get_requests(self):
        """Lazy load requests."""
        try:
            import requests
            return requests
        except ImportError:
            raise ImportError(
                "Could not import requests python package. "
                "Please install it with `pip install requests`."
            )

    def load_content_from_file(self, source: str) -> Union[str, Dict[str, Any]]:
        """Load content from a file or URL."""
        if self._is_url(source):
            requests = self._get_requests()
            try:
                response = requests.get(source, timeout=10)
                response.raise_for_status()
                html = response.text
            except requests.RequestException as e:
                raise Exception(f"Failed to fetch URL: {e}")
        else:
            if not self.can_handle(source):
                raise ValueError(f"Unsupported file type: {source}")
            try:
                with open(source, 'r', encoding='utf-8') as f:
                    html = f.read()
            except IOError as e:
                raise Exception(f"Failed to read file: {e}")
                
        return self._process_html(html)

    def load_content_from_stream(self, stream: Union[BytesIO, str]) -> Union[str, Dict[str, Any]]:
        """Load content from a stream."""
        if isinstance(stream, BytesIO):
            html = stream.read().decode('utf-8')
        else:
            html = stream
        return self._process_html(html)

    def _truncate_to_token_limit(self, text: str, max_tokens: int = 1000) -> str:
        """
        Truncates text to stay within a specified token limit, attempting to break at sentence boundaries.
        
        Args:
            text (str): The text to truncate
            max_tokens (int): Maximum number of tokens allowed (default: 1000)
        
        Returns:
            str: Truncated text
        """
        if not text:
            return text
            
        current_tokens = num_tokens_from_string(text)
        
        if current_tokens <= max_tokens:
            return text
            
        # Binary search for appropriate truncation point
        left, right = 0, len(text)
        while left < right:
            mid = (left + right) // 2
            
            potential_break = text.rfind('.', left, mid)
            if potential_break == -1:
                potential_break = mid
                
            truncated = text[:potential_break + 1]
            if num_tokens_from_string(truncated) <= max_tokens:
                left = mid + 1
            else:
                right = mid
                
        final_break = text.rfind('.', 0, left)
        if final_break == -1:
            final_break = left
            
        return text[:final_break + 1] + "..."

    def _process_html(self, html: str) -> Union[str, Dict[str, Any]]:
        """Process HTML content based on header handling strategy."""
        BeautifulSoup = self._get_bs4()
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer']):
            element.decompose()
            
        # Handle headers based on strategy
        headers = []
        if self.header_handling != "skip":
            for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                headers.append(header.get_text().strip())
                if self.header_handling == "summarize":
                    header.decompose()
        
        # Extract and clean text
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Apply token limit
        text = self._truncate_to_token_limit(text, max_tokens=1000)
        
        if self.header_handling == "summarize":
            return {
                "content": text,
                # "headers": headers,
                # "metadata": {
                #     "title": soup.title.string if soup.title else None,
                #     "url": soup.find('meta', {'property': 'og:url'}).get('content') if soup.find('meta', {'property': 'og:url'}) else None,
                #     "description": soup.find('meta', {'name': 'description'}).get('content') if soup.find('meta', {'name': 'description'}) else None
                # }
            }
        
        return text

    @staticmethod
    def _is_url(source: str) -> bool:
        """Check if the source is a URL."""
        try:
            result = urlparse(source)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def load_content_from_file_list(self, file_paths: List[str]) -> List[Union[str, Dict[str, Any]]]:
        """Load content from multiple files."""
        return [self.load_content_from_file(path) for path in file_paths]

    def load_content_from_stream_list(self, streams: List[Union[BytesIO, str]]) -> List[Union[str, Dict[str, Any]]]:
        """Load content from multiple streams."""
        return [self.load_content_from_stream(stream) for stream in streams]
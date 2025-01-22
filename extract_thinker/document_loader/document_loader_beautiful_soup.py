from typing import List, Union, Dict, Any, Optional
from io import BytesIO
from urllib.parse import urlparse
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader
from extract_thinker.utils import get_file_extension, num_tokens_from_string
from operator import attrgetter
from cachetools import cachedmethod
from cachetools.keys import hashkey
from dataclasses import dataclass, field


@dataclass
class BeautifulSoupConfig:
    """Configuration for BeautifulSoup HTML loader.
    
    Args:
        header_handling: How to handle headers - "skip", "summarize", or "include"
        content: Initial content (optional)
        cache_ttl: Cache time-to-live in seconds (default: 300)
        max_tokens: Maximum number of tokens per page (default: 1000)
        request_timeout: Timeout for HTTP requests in seconds (default: 10)
        parser: HTML parser to use (default: 'html.parser')
        remove_elements: List of HTML elements to remove (default: ['script', 'style', 'nav', 'footer'])
    """
    # Required parameters
    header_handling: str = "skip"
    
    # Optional parameters
    content: Optional[Any] = None
    cache_ttl: int = 300
    max_tokens: int = 1000
    request_timeout: int = 10
    parser: str = "html.parser"
    remove_elements: List[str] = field(default_factory=lambda: ['script', 'style', 'nav', 'footer'])

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_header_handling = {"skip", "summarize", "include"}
        if self.header_handling not in valid_header_handling:
            raise ValueError(
                f"Invalid header_handling value: {self.header_handling}. "
                f"Valid values are: {sorted(valid_header_handling)}"
            )
        
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
            
        if self.request_timeout < 1:
            raise ValueError("request_timeout must be positive")


class DocumentLoaderBeautifulSoup(CachedDocumentLoader):
    """Loader that uses BeautifulSoup4 to load HTML content."""
    
    SUPPORTED_FORMATS = ['html', 'htm']
    
    def __init__(
        self,
        header_handling_or_config: Union[str, BeautifulSoupConfig] = "skip",
        content: Optional[Any] = None,
        cache_ttl: int = 300,
        max_tokens: int = 1000,
        request_timeout: int = 10,
        parser: str = "html.parser",
        remove_elements: Optional[List[str]] = None
    ):
        """Initialize loader.
        
        Args:
            header_handling_or_config: Either a BeautifulSoupConfig object or header handling strategy
            content: Initial content (only used if header_handling_or_config is a string)
            cache_ttl: Cache time-to-live in seconds (only used if header_handling_or_config is a string)
            max_tokens: Maximum number of tokens per page (only used if header_handling_or_config is a string)
            request_timeout: Timeout for HTTP requests in seconds (only used if header_handling_or_config is a string)
            parser: HTML parser to use (only used if header_handling_or_config is a string)
            remove_elements: List of HTML elements to remove (only used if header_handling_or_config is a string)
        """
        # Check required dependencies
        self._check_dependencies()
        
        # Handle both config-based and old-style initialization
        if isinstance(header_handling_or_config, BeautifulSoupConfig):
            self.config = header_handling_or_config
        else:
            # Create config from individual parameters
            self.config = BeautifulSoupConfig(
                header_handling=header_handling_or_config,
                content=content,
                cache_ttl=cache_ttl,
                max_tokens=max_tokens,
                request_timeout=request_timeout,
                parser=parser,
                remove_elements=remove_elements or ['script', 'style', 'nav', 'footer']
            )
        
        super().__init__(self.config.content, self.config.cache_ttl)
        self.header_handling = self.config.header_handling

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
        soup = BeautifulSoup(html, self.config.parser)
        
        # Remove specified elements
        for element in soup(self.config.remove_elements):
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
        text = self._truncate_to_token_limit(text, max_tokens=self.config.max_tokens)
        
        if self.header_handling == "summarize":
            return {
                "content": text,
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

    @cachedmethod(cache=attrgetter('cache'), 
                  key=lambda self, source: hashkey(source if isinstance(source, str) else source.getvalue(), self.vision_mode))
    def load(self, source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        """
        Load content from a web page or HTML file and convert it to our standard format.
        
        Args:
            source: Either a URL, file path, or BytesIO stream
        
        Returns:
            List[Dict[str, Any]]: List of pages with content
        """
        if not self.can_handle(source):
            raise ValueError(f"Cannot handle source: {source}")

        # If in vision mode and can't handle vision, raise ValueError
        if self.vision_mode and not self.can_handle_vision(source):
            raise ValueError(f"Cannot handle source in vision mode: {source}")

        try:
            # Process based on source type
            if isinstance(source, str):
                if self._is_url(source):
                    requests = self._get_requests()
                    response = requests.get(source, timeout=10)
                    response.raise_for_status()
                    html = response.text
                else:
                    with open(source, 'r', encoding='utf-8') as f:
                        html = f.read()
            else:
                html = source.read().decode('utf-8')
                
            content = self._process_html(html)
                
            # Convert to standard page-based format
            if isinstance(content, dict):
                return [content]
            else:
                return [{"content": content}]
                
        except Exception as e:
            raise ValueError(f"Error loading HTML content: {str(e)}")

    def can_handle(self, source: Union[str, BytesIO]) -> bool:
        """Check if the loader can handle this source."""
        if isinstance(source, BytesIO):
            return True
        if self._is_url(source):
            return True
        return get_file_extension(source) in self.SUPPORTED_FORMATS
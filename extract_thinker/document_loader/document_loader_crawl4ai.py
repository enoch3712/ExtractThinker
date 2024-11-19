from typing import List, Union, Dict, Any
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader
from io import BytesIO
from urllib.parse import urlparse
import asyncio
import nest_asyncio
nest_asyncio.apply() 

from extract_thinker.utils import num_tokens_from_string

class DocumentLoaderCrawl4Ai(CachedDocumentLoader):
    """Loader that uses crawl4ai's AsyncWebCrawler for web content extraction."""
    
    SUPPORTED_FORMATS = ['html', 'htm']
    
    def __init__(self, header_handling: str = "skip", content: Any = None, cache_ttl: int = 300):
        super().__init__(content, cache_ttl)
        self.header_handling = header_handling
        try:
            self.crawler = AsyncWebCrawler()
        except Exception as e:
            print(f"Error initializing AsyncWebCrawler: {str(e)}")
            raise RuntimeError("Failed to initialize AsyncWebCrawler. Please ensure crawl4ai is properly installed and configured.") from e

    async def _analyze_page(self, url: str) -> Dict[str, Any]:
        """Analyze page content using AsyncWebCrawler."""
        try:
            # Create a new context for each run
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(
                    url=url,
                    bypass_cache=True,
                    exclude_external_links=False,
                    exclude_social_media_links=True
                )
                
                # Validate result
                if not result or not hasattr(result, 'html') or not result.html:
                    raise ValueError(f"Failed to get valid HTML content from {url}")
                    
                return {
                    'html': result.html,
                    'links': getattr(result, 'links', {'internal': [], 'external': []}),
                    'text': result.markdown,  # Using markdown instead of raw text
                    'metadata': {
                        'title': '',
                        'meta_description': '',
                        'meta_keywords': '',
                        'canonical_url': url
                    }
                }
        except Exception as e:
            print(f"Error analyzing {url}: {str(e)}")
            return {
                'html': '',
                'links': {'internal': [], 'external': []},
                'text': f'Error fetching content: {str(e)}',
                'metadata': {
                    'title': '',
                    'meta_description': '',
                    'meta_keywords': '',
                    'canonical_url': url
                }
            }

    def _truncate_to_token_limit(text: str, max_tokens: int = 1000) -> str:
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
            
        # Get initial token count
        current_tokens = num_tokens_from_string(text)
        
        if current_tokens <= max_tokens:
            return text
            
        # Binary search for appropriate truncation point
        left, right = 0, len(text)
        while left < right:
            mid = (left + right) // 2
            
            # Try to find a sentence boundary near the midpoint
            potential_break = text.rfind('.', left, mid)
            if potential_break == -1:
                potential_break = mid
                
            # Check token count of truncated text
            truncated = text[:potential_break + 1]
            if num_tokens_from_string(truncated) <= max_tokens:
                left = mid + 1
            else:
                right = mid
                
        # Find the last sentence boundary before the truncation point
        final_break = text.rfind('.', 0, left)
        if final_break == -1:
            final_break = left
            
        return text[:final_break + 1] + "..."


    def _process_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Process the content and return simplified format with content and buttons."""
        # Extract all button text from internal and external links
        all_links = content['links']['internal'] + content['links']['external']
        buttons = [
            {
                'text': link['text'].strip(),
                'href': link['href']
            }
            for link in all_links 
            if link['text'].strip()
        ]
        
        # Truncate content to 1000 tokens using tiktoken
        text_content = self._truncate_to_token_limit(content['text'], max_tokens=1000)
        
        return {
            "content": text_content,
            "buttons": buttons
        }

    def _classify_elements(self, links: Dict[str, List[Dict]]) -> Dict[str, List]:
        """Classify elements into different categories."""
        indicators = {
            "login_elements": [],
            "pricing_elements": [],
            "cta_buttons": [],
            "navigation_items": []
        }
        
        login_patterns = ['login', 'sign in', 'signin', 'log in', 'account', 'dashboard']
        pricing_patterns = ['pricing', 'plans', 'subscribe', 'trial', 'get started']
        
        # Process both internal and external links
        all_links = links['internal'] + links['external']
        
        for link in all_links:
            text = link['text'].lower()
            href = link['href'].lower()
            
            if any(pattern in text or pattern in href for pattern in login_patterns):
                indicators['login_elements'].append(link)
            
            if any(pattern in text or pattern in href for pattern in pricing_patterns):
                indicators['pricing_elements'].append(link)
            
            # Classify navigation items based on internal links
            if link in links['internal']:
                indicators['navigation_items'].append(link)
        
        return indicators

    def load_content_from_file(self, source: str) -> Union[str, Dict[str, Any]]:
        """Load content from a URL or file."""
        if source in self.cache:
            return self.cache[source]

        if self._is_url(source):
            content = asyncio.run(self._analyze_page(source))
            result = self._process_content(content)
        else:
            # Handle local files
            with open(source, 'r', encoding='utf-8') as f:
                html = f.read()
            soup = BeautifulSoup(html, 'html.parser')
            result = self._process_content({
                'html': html,
                'text': soup.get_text(),
                'links': {'internal': [], 'external': []},
                'metadata': {}
            })

        self.cache[source] = result
        return result

    def load_content_from_stream(self, stream: BytesIO) -> Union[str, Dict[str, Any]]:
        """Load content from a stream."""
        html = stream.read().decode('utf-8')
        soup = BeautifulSoup(html, 'html.parser')
        return self._process_content({
            'html': html,
            'text': soup.get_text(),
            'links': {'internal': [], 'external': []},
            'metadata': {}
        })

    def load_content_from_stream_list(self, streams: List[BytesIO]) -> List[Union[str, Dict[str, Any]]]:
        """Load content from multiple streams."""
        return [self.load_content_from_stream(stream) for stream in streams]

    def load_content_from_file_list(self, file_paths: List[str]) -> List[Union[str, Dict[str, Any]]]:
        """Load content from multiple files."""
        return [self.load_content_from_file(path) for path in file_paths]

    @staticmethod
    def _is_url(source: str) -> bool:
        """Check if the source is a URL."""
        try:
            result = urlparse(source)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False 
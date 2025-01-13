import io
from typing import Any, Union, Dict, List, Optional
from io import BytesIO
from operator import attrgetter
from cachetools import cachedmethod
from cachetools.keys import hashkey
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader
from dataclasses import dataclass


@dataclass
class PyPDFConfig:
    """Configuration for PyPDF loader.
    
    Args:
        content: Initial content (optional)
        cache_ttl: Cache time-to-live in seconds (default: 300)
        vision_enabled: Whether to enable vision mode for image extraction (default: False)
        password: Password for encrypted PDFs (optional)
        extract_text: Whether to extract text from the PDF (default: True)
    """
    content: Optional[Any] = None
    cache_ttl: int = 300
    vision_enabled: bool = False
    password: Optional[str] = None
    extract_text: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.cache_ttl <= 0:
            raise ValueError("cache_ttl must be positive")
        
        if self.password is not None and not isinstance(self.password, str):
            raise ValueError("password must be a string")


class DocumentLoaderPyPdf(CachedDocumentLoader):
    """Loader for PDFs using PyPDF (pypdf) to extract text, and pypdfium2 to extract images if vision mode is enabled."""
    SUPPORTED_FORMATS = ['pdf']
    
    def __init__(
        self,
        content_or_config: Union[Any, PyPDFConfig] = None,
        cache_ttl: int = 300,
        vision_enabled: bool = False,
        password: Optional[str] = None,
        extract_text: bool = True
    ):
        """Initialize loader.
        
        Args:
            content_or_config: Either a PyPDFConfig object or initial content
            cache_ttl: Cache time-to-live in seconds (only used if content_or_config is not PyPDFConfig)
            vision_enabled: Whether to enable vision mode (only used if content_or_config is not PyPDFConfig)
            password: Password for encrypted PDFs (only used if content_or_config is not PyPDFConfig)
            extract_text: Whether to extract text (only used if content_or_config is not PyPDFConfig)
        """
        # Check required dependencies
        self._check_dependencies()

        # Handle both config-based and old-style initialization
        if isinstance(content_or_config, PyPDFConfig):
            self.config = content_or_config
        else:
            # Create config from individual parameters
            self.config = PyPDFConfig(
                content=content_or_config,
                cache_ttl=cache_ttl,
                vision_enabled=vision_enabled,
                password=password,
                extract_text=extract_text
            )
        
        super().__init__(self.config.content, self.config.cache_ttl)
        self.vision_mode = self.config.vision_enabled

    def set_vision_mode(self, enabled: bool = True):
        """Enable or disable vision mode."""
        self.vision_mode = enabled
        self.config.vision_enabled = enabled

    @staticmethod
    def _check_dependencies():
        """Check if required dependencies are installed."""
        try:
            import pypdf
        except ImportError:
            raise ImportError(
                "Could not import pypdf python package. "
                "Please install it with `pip install pypdf`."
            )

    def _get_pypdf(self):
        """Lazy load pypdf."""
        try:
            from pypdf import PdfReader
            return PdfReader
        except ImportError:
            raise ImportError(
                "Could not import pypdf python package. "
                "Please install it with `pip install pypdf`."
            )

    @cachedmethod(cache=attrgetter('cache'), 
                  key=lambda self, source: hashkey(source if isinstance(source, str) else source.getvalue(), self.vision_mode))
    def load(self, source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        """
        Load the PDF from a file path or a BytesIO stream.
        Return a list of dictionaries, each representing one page:
          - "content": extracted text (if extract_text is True)
          - "image": (bytes) rendered image of the page if vision_mode is True
        """
        if not self.can_handle(source):
            raise ValueError(f"Cannot handle source: {source}")

        PdfReader = self._get_pypdf()

        try:
            # Read the PDF pages using pypdf
            if isinstance(source, str):
                reader = PdfReader(source, password=self.config.password)
            else:
                # BytesIO
                source.seek(0)
                reader = PdfReader(source, password=self.config.password)

            pages_data = []
            num_pages = len(reader.pages)

            # Extract text for each page if enabled
            for page_index in range(num_pages):
                pdf_page = reader.pages[page_index]
                page_dict = {}
                
                if self.config.extract_text:
                    page_dict["content"] = pdf_page.extract_text() or ''
                else:
                    page_dict["content"] = ""

                pages_data.append(page_dict)

            # If vision_mode is enabled, convert entire PDF into a dictionary of images
            if self.vision_mode:
                images_dict = self.convert_to_images(source)
                for idx, page_dict in enumerate(pages_data):
                    if idx in images_dict:
                        page_dict["image"] = images_dict[idx]

            return pages_data
            
        except Exception as e:
            raise ValueError(f"Error loading PDF: {str(e)}")

    def can_handle_vision(self, source: Union[str, io.BytesIO]) -> bool:
        """Check if this loader can handle the source in vision mode."""
        return self.config.vision_enabled and self.can_handle(source)
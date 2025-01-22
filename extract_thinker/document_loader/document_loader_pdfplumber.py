import io
from typing import Any, Union, Dict, List, Optional
from io import BytesIO
from operator import attrgetter
from cachetools import cachedmethod
from cachetools.keys import hashkey
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader
from dataclasses import dataclass, field


@dataclass
class PDFPlumberConfig:
    """Configuration for PDFPlumber loader.
    
    Args:
        content: Initial content (optional)
        cache_ttl: Cache time-to-live in seconds (default: 300)
        table_settings: Custom settings for table extraction (optional)
        vision_enabled: Whether to enable vision mode for image extraction (default: False)
        extract_tables: Whether to extract tables from the PDF (default: True)
    """
    content: Optional[Any] = None
    cache_ttl: int = 300
    table_settings: Optional[Dict[str, Any]] = None
    vision_enabled: bool = False
    extract_tables: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.cache_ttl <= 0:
            raise ValueError("cache_ttl must be positive")
        
        if self.table_settings is not None and not isinstance(self.table_settings, dict):
            raise ValueError("table_settings must be a dictionary")


class DocumentLoaderPdfPlumber(CachedDocumentLoader):
    """Loader for PDFs using pdfplumber, supporting text and table extraction."""
    SUPPORTED_FORMATS = ['pdf']

    def __init__(
        self,
        content_or_config: Union[Any, PDFPlumberConfig] = None,
        cache_ttl: int = 300,
        table_settings: Optional[Dict[str, Any]] = None,
        vision_enabled: bool = False,
        extract_tables: bool = True
    ):
        """Initialize loader.
        
        Args:
            content_or_config: Either a PDFPlumberConfig object or initial content
            cache_ttl: Cache time-to-live in seconds (only used if content_or_config is not PDFPlumberConfig)
            table_settings: Custom settings for table extraction (only used if content_or_config is not PDFPlumberConfig)
            vision_enabled: Whether to enable vision mode (only used if content_or_config is not PDFPlumberConfig)
            extract_tables: Whether to extract tables (only used if content_or_config is not PDFPlumberConfig)
        """
        # Check required dependencies
        self._check_dependencies()

        # Handle both config-based and old-style initialization
        if isinstance(content_or_config, PDFPlumberConfig):
            self.config = content_or_config
        else:
            # Create config from individual parameters
            self.config = PDFPlumberConfig(
                content=content_or_config,
                cache_ttl=cache_ttl,
                table_settings=table_settings,
                vision_enabled=vision_enabled,
                extract_tables=extract_tables
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
            import pdfplumber
        except ImportError:
            raise ImportError(
                "Could not import pdfplumber python package. "
                "Please install it with `pip install pdfplumber`."
            )

    def _get_pdfplumber(self):
        """Lazy load pdfplumber."""
        try:
            import pdfplumber
            return pdfplumber
        except ImportError:
            raise ImportError(
                "Could not import pdfplumber python package. "
                "Please install it with `pip install pdfplumber`."
            )

    @cachedmethod(cache=attrgetter('cache'), 
                  key=lambda self, source: hashkey(source if isinstance(source, str) else source.getvalue(), self.vision_mode))
    def load(self, source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        """
        Load a PDF and extract text and tables from each page.
        Returns a list of pages, each containing:
        - content: The text content of the page
        - tables: Any tables found on the page
        - image: The rendered page image (if vision_mode is True)
        
        Args:
            source: Either a file path or BytesIO stream
            
        Returns:
            List[Dict[str, Any]]: List of pages with content and optional images
        """
        if not self.can_handle(source):
            raise ValueError(f"Cannot handle source: {source}")

        pdfplumber = self._get_pdfplumber()

        try:
            # Open PDF with pdfplumber
            if isinstance(source, str):
                pdf = pdfplumber.open(source)
            else:
                source.seek(0)
                pdf = pdfplumber.open(source)

            pages = []
            for page in pdf.pages:
                # Extract text and tables for this page
                page_dict = {
                    "content": page.extract_text() or "",
                    "tables": self._extract_tables(page)
                }
                
                # Add image if in vision mode
                if self.vision_mode:
                    images_dict = self.convert_to_images(source)
                    page_number = page.page_number - 1
                    if page_number in images_dict:
                        page_dict["image"] = images_dict[page_number]
                
                pages.append(page_dict)

            pdf.close()
            return pages

        except Exception as e:
            raise ValueError(f"Error loading PDF: {str(e)}")

    def _extract_tables(self, page) -> List[List[List[str]]]:
        """Extract and clean tables from a page."""
        if not self.config.extract_tables:
            return []

        tables = []
        try:
            # Use custom table settings if provided, otherwise use defaults
            settings = self.config.table_settings if self.config.table_settings else {
                'vertical_strategy': 'text',
                'horizontal_strategy': 'text',
                'intersection_y_tolerance': 10,
                'intersection_x_tolerance': 10
            }
            
            extracted_tables = page.extract_tables(settings)

            # Clean and process tables
            for table in extracted_tables:
                if not table:
                    continue
                    
                # Clean table data
                cleaned_table = [
                    [str(cell).strip() if cell is not None else "" for cell in row]
                    for row in table
                ]
                
                # Remove empty rows
                cleaned_table = [
                    row for row in cleaned_table 
                    if any(cell != "" for cell in row)
                ]
                
                if cleaned_table:
                    tables.append(cleaned_table)
                    
            return tables
            
        except Exception as e:
            print(f"Warning: Error extracting tables: {str(e)}")
            return []

    def can_handle_vision(self, source: Union[str, io.BytesIO]) -> bool:
        """Check if this loader can handle the source in vision mode."""
        return self.config.vision_enabled and self.can_handle(source)
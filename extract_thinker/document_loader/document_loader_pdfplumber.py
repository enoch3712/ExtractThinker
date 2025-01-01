import io
from typing import Any, Union, Dict, List
from io import BytesIO
from operator import attrgetter
from cachetools import cachedmethod
from cachetools.keys import hashkey
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader

class DocumentLoaderPdfPlumber(CachedDocumentLoader):
    """Loader for PDFs using pdfplumber, supporting text and table extraction."""
    SUPPORTED_FORMATS = ['pdf']

    def __init__(self, content: Any = None, cache_ttl: int = 300):
        """Initialize loader.
        
        Args:
            content: Initial content
            cache_ttl: Cache time-to-live in seconds
        """
        # Check required dependencies
        self._check_dependencies()
        super().__init__(content, cache_ttl)

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
        """Extract and clean tables from a page.
        
        Args:
            page: PDFPlumber page object
            
        Returns:
            List of tables, where each table is a list of rows
        """
        tables = []
        try:
            # Extract tables with default settings
            extracted_tables = page.extract_tables()
            
            if not extracted_tables:
                # Try with different settings if no tables found
                settings = {
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
                    [
                        str(cell).strip() if cell is not None else ""
                        for cell in row
                    ]
                    for row in table
                ]
                
                # Remove empty rows and columns
                cleaned_table = [
                    row for row in cleaned_table 
                    if any(cell != "" for cell in row)
                ]
                
                if cleaned_table:
                    tables.append(cleaned_table)
                    
            return tables
            
        except Exception as e:
            # Log error but continue processing
            print(f"Warning: Error extracting tables: {str(e)}")
            return []

    def can_handle_vision(self, source: Union[str, io.BytesIO]) -> bool:
        """Check if this loader can handle the source in vision mode."""
        return self.can_handle(source)
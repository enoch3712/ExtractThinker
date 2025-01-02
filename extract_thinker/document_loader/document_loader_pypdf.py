import io
from typing import Any, Union
from io import BytesIO
from operator import attrgetter
from cachetools import cachedmethod
from cachetools.keys import hashkey
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader

class DocumentLoaderPyPdf(CachedDocumentLoader):
    """Loader for PDFs using PyPDF (pypdf) to extract text, and pypdfium2 to extract images if vision mode is enabled."""
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
    def load(self, source: Union[str, BytesIO]) -> Any:
        """
        Load the PDF from a file path or a BytesIO stream.
        Return a list of dictionaries, each representing one page:
          - "content": extracted text
          - "image": (bytes) rendered image of the page if vision_mode is True
        """
        if not self.can_handle(source):
            raise ValueError(f"Cannot handle source: {source}")

        PdfReader = self._get_pypdf()

        # Read the PDF pages using pypdf
        if isinstance(source, str):
            reader = PdfReader(source)
        else:
            # BytesIO
            source.seek(0)
            reader = PdfReader(source)

        pages_data = []
        num_pages = len(reader.pages)

        # Extract text for each page
        for page_index in range(num_pages):
            pdf_page = reader.pages[page_index]
            page_text = pdf_page.extract_text() or ''
            page_dict = {"content": page_text}
            pages_data.append(page_dict)

        # If vision_mode is enabled, convert entire PDF into a dictionary of images keyed by page number
        # and attach them to the corresponding page dictionary
        if self.vision_mode:
            # Use the same approach as in base class: convert_to_images
            # This tries PDF conversion if it's not a raw image
            images_dict = self.convert_to_images(source)
            # images_dict is a dict {page_index: image_bytes}
            for idx, page_dict in enumerate(pages_data):
                if idx in images_dict:
                    page_dict["image"] = images_dict[idx]

        return pages_data
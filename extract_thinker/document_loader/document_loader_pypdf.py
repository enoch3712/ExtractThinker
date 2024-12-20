from typing import Any, Union
from io import BytesIO
from pypdf import PdfReader
from operator import attrgetter
from cachetools import cachedmethod
from cachetools.keys import hashkey
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader

class DocumentLoaderPyPdf(CachedDocumentLoader):
    """Loader for PDFs using PyPDF (pypdf) to extract text, and pypdfium2 to extract images if vision mode is enabled."""
    SUPPORTED_FORMATS = ['pdf']

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

            # If vision_mode is enabled, convert page to an image
            if self.vision_mode:
                # We'll rely on convert_to_images (inherited from DocumentLoader),
                # which calls pypdfium2 to render each page as images (page-index keyed)
                pass

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
from io import BytesIO
from typing import Any, Dict, List, Union, Optional

from cachetools import cachedmethod
from cachetools.keys import hashkey

# Import your DocumentLoader base
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader

class DocumentLoaderDocling(CachedDocumentLoader):
    """
    Document loader that uses Docling to extract content from various file formats.
    Produces a list of pages, each with:
      - "content": text from that page
      - "image": optional page image bytes if vision_mode is True
      - "markdown": Markdown string of that page
    """

    SUPPORTED_FORMATS = [
        # Microsoft Word family
        "docx", "dotx", "docm", "dotm",
        # Microsoft PowerPoint family
        "pptx", "potx", "ppsx", "pptm", "potm", "ppsm",
        # Microsoft Excel family
        "xlsx",
        # PDF
        "pdf",
        # HTML variants
        "html", "htm", "xhtml",
        # Markdown
        "md",
        # AsciiDoc variants
        "adoc", "asciidoc", "asc",
        # Common image types
        "png", "jpg", "jpeg", "tif", "tiff", "bmp",
        # XML (including PubMed .nxml)
        "xml", "nxml",
        # Plain text
        "txt"
    ]

    def __init__(
        self,
        content: Any = None,
        cache_ttl: int = 300,
        format_options: Optional[Dict[str, Any]] = None,
    ):
        """Initialize loader.
        
        Args:
            content: Initial content
            cache_ttl: Cache time-to-live in seconds
            format_options: Dictionary mapping input formats to their FormatOption configurations
                Example:
                {
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
                    InputFormat.IMAGE: ImageFormatOption(pipeline_options=image_options),
                    ...
                }
        """
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import FormatOption

        # Check dependencies before any initialization
        self._check_dependencies()
        
        super().__init__(content, cache_ttl)
        self.format_options = format_options
        self.converter = self._init_docling_converter()

    @staticmethod
    def _check_dependencies():
        """Check if required dependencies are installed."""
        try:
            import docling
            import docling.document_converter
            import docling.datamodel.document
            import docling.datamodel.pipeline_options
            import docling_core.types.doc
        except ImportError:
            raise ImportError(
                "Could not import docling python package. "
                "Please install it with `pip install docling`."
            )

    def _init_docling_converter(self):
        """Initialize the Docling document converter."""
        from docling.document_converter import DocumentConverter
        return DocumentConverter()

    @cachedmethod(cache=lambda self: self.cache, 
                  key=lambda self, source: hashkey(
                      source if isinstance(source, str) else source.getvalue(), 
                      self.vision_mode
                  ))
    def load(self, source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        """
        Load and parse the document using Docling.
        
        Returns:
            A list of dictionaries, each representing a "page" with:
              - "content": text from that page
              - "image": optional image bytes if vision_mode is True
              - "markdown": Markdown string of that page
        """
        if not self.can_handle(source):
            raise ValueError(f"Cannot handle source: {source}")

        # Convert the source to a docling "ConversionResult"
        conv_result = self._docling_convert(source)

        test = conv_result.document.export_to_markdown()
        print(test)
        
        # Build the output list of page data
        pages_output = []
        for p in conv_result.pages:
            page_dict = {
                "content": conv_result.document.export_to_markdown(page_no=p.page_no+1),
                "image": None
            }

            # Handle image extraction if vision_mode is enabled
            if self.vision_mode:
                images_dict = self.convert_to_images(source)
                page_dict["image"] = images_dict.get(p.page_no)

            pages_output.append(page_dict)

        # Fallback for documents without explicit pages
        if not pages_output:
            doc_text = conv_result.document.export_to_markdown()
            pages_output = [{"content": doc_text, "image": None}]

        return pages_output

    def _docling_convert(self, source: Union[str, BytesIO]) -> Any:
        """
        Internal method that runs the docling convert pipeline.
        Uses format_options if provided during initialization, otherwise uses default settings.
        """
        from docling.document_converter import DocumentConverter
        from docling_core.types.io import DocumentStream
        import uuid
        
        # Create converter with optional format options from initialization
        docling_converter = DocumentConverter(
            format_options=self.format_options if self.format_options else None
        )

        # Handle different input types
        if isinstance(source, BytesIO):
            # Generate a unique filename using UUID
            unique_filename = f"{uuid.uuid4()}.pdf"
            doc_stream = DocumentStream(name=unique_filename, stream=source)
            conv_result = docling_converter.convert(doc_stream, raises_on_error=True)
        elif isinstance(source, str):
            # Handle string paths or URLs directly
            conv_result = docling_converter.convert(source, raises_on_error=True)
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        return conv_result

    def _extract_page_text(self, page: Any) -> str:
        """
        Gather text from a docling Page object. 
        Handles both text and table items.
        """

        from docling_core.types.doc import DocItemLabel, TableItem

        lines = []
        if page.assembled and page.assembled.elements:
            for element in page.assembled.elements:
                # Normal text
                if element.label in [DocItemLabel.TEXT, DocItemLabel.PARAGRAPH]:
                    lines.append(element.text or "")
                # Tables
                elif element.label == DocItemLabel.TABLE and isinstance(element, TableItem):
                    table_text = self.convert_table_to_text(element)
                    lines.append(table_text)
        else:
            # If no "assembled" data, fallback to the raw text cells 
            # or produce an empty string
            if page.cells:
                # Join cell texts. Not always great, but a fallback:
                return "\n".join(cell.text for cell in page.cells if cell.text)
            return ""

        return "\n".join(lines)

    def convert_table_to_text(self, table_item: Any) -> str:
        """
        Convert a TableItem to a Markdown table string.
        """
        headers = []
        rows = []

        # Assuming the first row is the header
        for idx, row in enumerate(table_item.table_rows):
            row_text = []
            for cell in row.table_cells:
                row_text.append(cell.text.strip() if cell.text else "")
            if idx == 0:
                headers = row_text
                rows.append("| " + " | ".join(headers) + " |")
                rows.append("| " + " | ".join(['---'] * len(headers)) + " |")
            else:
                rows.append("| " + " | ".join(row_text) + " |")

        return "\n".join(rows)


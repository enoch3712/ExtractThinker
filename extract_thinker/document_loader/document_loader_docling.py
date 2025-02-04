from io import BytesIO
from typing import Any, Dict, List, Union, Optional
from dataclasses import dataclass, field
from urllib.parse import urlparse

from cachetools import cachedmethod
from cachetools.keys import hashkey

from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader


@dataclass
class DoclingConfig:
    """Configuration for Docling document loader.
    
    This class supports both simple and complex configurations:
    
    Simple usage:
        config = DoclingConfig()  # Uses default settings
        
    Complex usage:
        config = DoclingConfig(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=PdfPipelineOptions(
                        do_table_structure=True,
                        do_ocr=True,
                        table_structure_options=TableStructureOptions(
                            do_cell_matching=True
                        )
                    )
                )
            }
        )
    
    Args:
        content: Initial content (optional)
        cache_ttl: Cache time-to-live in seconds (default: 300)
        format_options: Dictionary mapping input formats to their FormatOption configurations.
            If None, default options will be created based on other parameters.
            For complex scenarios, you can provide your own format options:
            {
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
                InputFormat.IMAGE: ImageFormatOption(pipeline_options=image_options),
                ...
            }
        ocr_enabled: Whether to enable OCR processing (default: False)
        table_structure_enabled: Whether to enable table structure detection (default: True)
        force_full_page_ocr: Whether to force OCR on entire pages (default: False)
        do_cell_matching: Whether to enable cell matching in tables (default: True)
    """
    # Optional parameters
    content: Optional[Any] = None
    cache_ttl: int = 300
    format_options: Optional[Dict[str, Any]] = None
    ocr_enabled: bool = False  # OCR disabled by default
    table_structure_enabled: bool = True
    force_full_page_ocr: bool = False
    do_cell_matching: bool = True

    def __post_init__(self):
        """Initialize format options if not provided."""
        # If format_options are provided, use them as is (complex configuration)
        if self.format_options is not None:
            return

        # Simple configuration: create default format options based on parameters
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            TableStructureOptions,
        )
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import PdfFormatOption

        # Set up table options
        table_options = None
        if self.table_structure_enabled:
            table_options = TableStructureOptions(
                do_cell_matching=self.do_cell_matching
            )

        # Create pipeline options
        pipeline_options = PdfPipelineOptions(
            do_table_structure=self.table_structure_enabled,
            do_ocr=self.ocr_enabled,
            table_structure_options=table_options
        )

        # Create format options
        self.format_options = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }


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
        "txt",
        # URL support
        "url"
    ]

    def __init__(
        self,
        content_or_config: Union[Any, DoclingConfig] = None,
        cache_ttl: int = 300,
        format_options: Optional[Dict[str, Any]] = None,
    ):
        """Initialize loader.
        
        Args:
            content_or_config: Either a DoclingConfig object or initial content
            cache_ttl: Cache time-to-live in seconds (only used if content_or_config is not DoclingConfig)
            format_options: Dictionary mapping input formats to their FormatOption configurations
                (only used if content_or_config is not DoclingConfig)
        """
        # Check dependencies before any initialization
        self._check_dependencies()
        
        # Handle both config-based and old-style initialization
        if isinstance(content_or_config, DoclingConfig):
            self.config = content_or_config
        else:
            # Create config from individual parameters
            self.config = DoclingConfig(
                content=content_or_config,
                cache_ttl=cache_ttl,
                format_options=format_options
            )
        
        super().__init__(self.config.content, self.config.cache_ttl)
        self.format_options = self.config.format_options
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

    def _is_url(self, potential_url: str) -> bool:
        """
        Check if the given string is a URL.
        
        Returns:
            True if the string starts with "http://" or "https://", otherwise False.
        """
        return potential_url.startswith("http://") or potential_url.startswith("https://")

    def can_handle(self, source: Union[str, BytesIO]) -> bool:
        """
        Determine if the loader can handle the given source.
        This method now supports URLs, local file paths with supported extensions, and BytesIO.
        
        Args:
            source: The document source, which may be a string (file path or URL) or a BytesIO stream.
            
        Returns:
            True if the source is a valid input for the loader, else False.
        """
        if isinstance(source, BytesIO):
            return True
        elif isinstance(source, str):
            # If it's a URL, return True.
            if self._is_url(source):
                return True
            # Otherwise, determine the file extension and check if it's supported.
            extension = source.split('.')[-1].lower()
            return extension in self.SUPPORTED_FORMATS
        return False

    @cachedmethod(cache=lambda self: self.cache, 
                  key=lambda self, source: hashkey(
                      source if isinstance(source, str) else source.getvalue(), 
                      self.vision_mode
                  ))
    def load(self, source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        from docling.document_converter import ConversionResult
        """
        Load and parse the document using Docling.
        
        Returns:
            A list of dictionaries, each representing a "page" with:
              - "content": text from that page
              - "image": optional image bytes if vision_mode is True
        """
        if not self.can_handle(source):
            raise ValueError(f"Cannot handle source: {source}")

        # Convert the source to a docling "ConversionResult"
        conv_result: ConversionResult = self._docling_convert(source)
        
        # If the source is a URL, return a single page with all the content.
        if isinstance(source, str) and self._is_url(source):
            content = conv_result.document.export_to_markdown()
            print(content)  # Log the exported markdown, if needed
            page_output = {"content": content, "image": None}
            # Handle image extraction if vision_mode is enabled
            if self.vision_mode:
                images_dict = self.convert_to_images(source)
                page_output["images"] = images_dict.get(0)
            return [page_output]

        # Build the output list of page data for non-URL sources
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
                # Titles
                if element.label == DocItemLabel.TITLE:
                    lines.append(f"# {element.text or ''}")
                
                # Section headers
                elif element.label == DocItemLabel.SECTION_HEADER:
                    lines.append(f"## {element.text or ''}")
                
                # Code blocks
                elif element.label == DocItemLabel.CODE:
                    code_text = element.text or ""
                    lines.append(f"```\n{code_text}\n```")
                
                # List items
                elif element.label == DocItemLabel.LIST_ITEM:
                    lines.append(f"- {element.text or ''}")
                
                # Normal text and paragraphs
                elif element.label in [DocItemLabel.TEXT, DocItemLabel.PARAGRAPH]:
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
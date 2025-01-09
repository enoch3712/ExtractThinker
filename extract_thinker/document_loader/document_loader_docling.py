from io import BytesIO
from typing import Any, Dict, List, Union
import logging

from cachetools import cachedmethod
from cachetools.keys import hashkey

# Import your DocumentLoader base
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader

# Docling imports
from docling.document_converter import DocumentConverter
from docling.datamodel.document import ConversionResult, Page
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import DocItemLabel, TableItem, DoclingDocument

logger = logging.getLogger(__name__)

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
        # Example: let the user pass in pipeline options 
        # or fallback to docling defaults
        pdf_pipeline_options: PdfPipelineOptions = None
    ):
        super().__init__(content, cache_ttl)
        self.converter = DocumentConverter()
        
        # Optionally store a pipeline config for PDF or other formats
        self.pdf_pipeline_options = pdf_pipeline_options
        if not self.pdf_pipeline_options:
            # Enable table structure extraction
            self.pdf_pipeline_options = PdfPipelineOptions(
                do_table_structure=True,
                do_ocr=False,
            )

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
            pages_output = [{"content": doc_text, "image": None, "markdown": doc_text}]

        return pages_output

    def _docling_convert(self, source: Union[str, BytesIO]) -> ConversionResult:
        """
        Internal method that runs the docling convert pipeline with OCR support.
        """
        from docling.document_converter import FormatOption, PdfFormatOption
        from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            TesseractCliOcrOptions
        )

        # Create pipeline options with OCR and table structure enabled
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True

        # Configure Tesseract OCR with full page mode
        ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True, tesseract_cmd="/opt/homebrew/bin/tesseract")
        pipeline_options.ocr_options = ocr_options

        # For PDF, override with pipeline options
        pdf_format_opt = PdfFormatOption(
            pipeline_options=pipeline_options,
            backend=DoclingParseV2DocumentBackend
        )

        # Create a converter with custom PDF format
        from_format_opts = {
            InputFormat.PDF: pdf_format_opt, 
            InputFormat.IMAGE: pdf_format_opt
        }
        
        docling_converter = DocumentConverter(format_options=from_format_opts)

        conv_result = docling_converter.convert(source, raises_on_error=True)

        return conv_result

    def _extract_page_text(self, page: Page) -> str:
        """
        Gather text from a docling Page object. 
        Handles both text and table items.
        """
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

    def convert_table_to_text(self, table_item: TableItem) -> str:
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


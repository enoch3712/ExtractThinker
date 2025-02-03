import os
import pytest
from extract_thinker.document_loader.document_loader_docling import DocumentLoaderDocling, DoclingConfig
from tests.test_document_loader_base import BaseDocumentLoaderTest
from io import BytesIO

from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,
    TableStructureOptions,
)
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption


class TestDocumentLoaderDocling(BaseDocumentLoaderTest):
    @pytest.fixture
    def test_file_path(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, 'files', 'invoice.pdf')

    @pytest.fixture
    def loader(self):
        """Required fixture from BaseDocumentLoaderTest - returns a basic loader instance"""
        return DocumentLoaderDocling()

    @pytest.fixture
    def default_pipeline_options(self):
        """Default pipeline options for testing"""
        ocr_options = TesseractCliOcrOptions(
            force_full_page_ocr=True,
            tesseract_cmd="/opt/homebrew/bin/tesseract"
        )
        
        table_options = TableStructureOptions(
            do_cell_matching=True
        )
        
        return PdfPipelineOptions(
            do_table_structure=True,
            do_ocr=True,
            ocr_options=ocr_options,
            table_structure_options=table_options
        )

    def test_simple_initialization(self, test_file_path, loader):
        """Test simple initialization without any configuration"""
        # Basic load and verify
        pages = loader.load(test_file_path)
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert "content" in pages[0]
        assert isinstance(pages[0]["content"], str)
        assert len(pages[0]["content"]) > 0

    def test_simple_config(self, test_file_path):
        """Test simple configuration with basic options"""
        config = DoclingConfig(
            ocr_enabled=False,
            table_structure_enabled=True,
            do_cell_matching=True
        )
        loader = DocumentLoaderDocling(config)
        
        pages = loader.load(test_file_path)
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert "content" in pages[0]

    def test_complex_config(self, test_file_path):
        """Test complex configuration with custom format options"""
        # Set up pipeline options
        pipeline_options = PdfPipelineOptions(
            do_table_structure=True,
            do_ocr=False,
            table_structure_options=TableStructureOptions(
                do_cell_matching=True
            )
        )
        
        # Create format options
        format_options = {
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
        
        # Create config with format options
        config = DoclingConfig(format_options=format_options)
        loader = DocumentLoaderDocling(config)
        
        pages = loader.load(test_file_path)
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert "content" in pages[0]

    def test_stream_loading(self, test_file_path, loader):
        """Test loading from BytesIO stream"""
        with open(test_file_path, 'rb') as f:
            stream = BytesIO(f.read())
            pages = loader.load(stream)
            
            assert isinstance(pages, list)
            assert len(pages) > 0
            assert "content" in pages[0]

    def test_vision_mode(self, test_file_path, loader):
        """Test vision mode functionality"""
        loader.set_vision_mode(True)
        
        pages = loader.load(test_file_path)
        assert isinstance(pages, list)
        assert len(pages) > 0
        
        for page in pages:
            assert isinstance(page, dict)
            assert "content" in page
            if loader.can_handle_vision(test_file_path):
                assert "image" in page
                assert isinstance(page["image"], bytes)

    def test_pagination(self, test_file_path, loader):
        """Test pagination functionality"""
        pages = loader.load(test_file_path)
        assert isinstance(pages, list)
        if loader.can_handle_paginate(test_file_path):
            assert len(pages) > 0
            for page in pages:
                assert "content" in page
                assert isinstance(page["content"], str)

    def test_supported_formats(self, loader):
        """Test that supported formats are correctly defined"""
        assert isinstance(loader.SUPPORTED_FORMATS, list)
        assert "pdf" in loader.SUPPORTED_FORMATS
        assert "docx" in loader.SUPPORTED_FORMATS
        assert "txt" in loader.SUPPORTED_FORMATS

    def test_ocr_disabled(self, test_file_path):
        """Test that OCR is disabled by default"""
        config = DoclingConfig()  # Default config
        loader = DocumentLoaderDocling(config)
        
        pages = loader.load(test_file_path)
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert "content" in pages[0]

    def test_ocr_enabled(self, test_file_path, default_pipeline_options):
        """Test with OCR enabled using tesseract"""
        # Create format options with OCR
        format_options = {
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=default_pipeline_options
            )
        }
        
        config = DoclingConfig(
            format_options=format_options,
            ocr_enabled=True,
            force_full_page_ocr=True
        )
        loader = DocumentLoaderDocling(config)
        
        pages = loader.load(test_file_path)
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert "content" in pages[0]

    def test_custom_ocr_config(self, test_file_path):
        """Test with custom OCR configuration"""
        # Set up OCR options
        ocr_options = TesseractCliOcrOptions(
            force_full_page_ocr=True,
            tesseract_cmd="/opt/homebrew/bin/tesseract"
        )
        
        # Set up pipeline options with OCR
        pipeline_options = PdfPipelineOptions(
            do_table_structure=True,
            do_ocr=True,
            ocr_options=ocr_options,
            table_structure_options=TableStructureOptions(
                do_cell_matching=True
            )
        )
        
        # Create format options
        format_options = {
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
        
        # Create config with OCR enabled
        config = DoclingConfig(
            format_options=format_options,
            ocr_enabled=True,
            force_full_page_ocr=True
        )
        loader = DocumentLoaderDocling(config)
        
        pages = loader.load(test_file_path)
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert "content" in pages[0]

    def test_title_extraction(self):
        """
        Test that a PDF with a recognized Title actually shows that Title
        in the extracted text or markdown.
        """

        loader = DocumentLoaderDocling()

        # 1. Provide the path to your custom test file with a known Title
        current_dir = os.path.dirname(os.path.abspath(__file__))
        test_pdf_path = os.path.join(current_dir, 'files', 'fca-approach-payment-services-electronic-money-2017-5.pdf')

        # 2. Load it
        pages = loader.load(test_pdf_path)
        assert pages, "No pages were returned from the PDF."

        # 3. Inspect the text from the first page (or all pages)
        page_text = pages[0]["content"]  # or loop over pages if you prefer
        assert isinstance(page_text, str), "Expected 'content' to be a string."

        # 4. Check that your known Title text is present
        #    Suppose your PDF has "Document Title" as the Title.
        assert "## 1 ntroduction" in page_text, (
            "Expected the recognized Title ('1 Introduction') "
            "to appear in the extracted text."
        )

    def test_url_loading(self, loader):
        """Test loading from a URL for Docling loader."""
        url = "https://www.handbook.fca.org.uk/handbook/BCOBS/2/?view=chapter"
        # Ensure the loader recognizes and can handle a URL
        assert loader.can_handle(url) is True

        pages = loader.load(url)
        assert isinstance(pages, list)
        assert len(pages) > 0
        for page in pages:
            assert "content" in page
            assert isinstance(page["content"], str)
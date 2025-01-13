import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
from docling.document_converter import PdfFormatOption, ImageFormatOption

class TestDocumentLoaderDocling(BaseDocumentLoaderTest):
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

    @pytest.fixture
    def docling_config(self, default_pipeline_options):
        """Default Docling configuration for testing"""
        format_options = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=default_pipeline_options)
        }
        return DoclingConfig(
            format_options=format_options,
            ocr_enabled=True,
            table_structure_enabled=True,
            tesseract_cmd="/opt/homebrew/bin/tesseract",
            force_full_page_ocr=True,
            do_cell_matching=True
        )

    @pytest.fixture
    def loader(self, docling_config):
        return DocumentLoaderDocling(docling_config)

    @pytest.fixture
    def loader_no_ocr(self):
        """Loader instance with OCR disabled"""
        return DocumentLoaderDocling(
            DoclingConfig(
                ocr_enabled=False,
                table_structure_enabled=True
            )
        )

    @pytest.fixture
    def test_file_path(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, 'files', 'invoice.pdf')

    def test_docling_specific_content(self, loader, test_file_path):
        """Test Docling-specific content extraction"""
        pages = loader.load(test_file_path)
        
        assert isinstance(pages, list)
        assert len(pages) > 0
        
        first_page = pages[0]
        assert "content" in first_page
        assert len(first_page["content"]) > 0

    def test_vision_mode(self, loader, test_file_path):
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

    def test_stream_loading(self, loader, test_file_path):
        """Test loading from BytesIO stream"""
        with open(test_file_path, 'rb') as f:
            stream = BytesIO(f.read())
            pages = loader.load(stream)
            
            assert isinstance(pages, list)
            assert len(pages) > 0
            assert "content" in pages[0]

    def test_pagination(self, loader, test_file_path):
        """Test pagination functionality"""
        pages = loader.load(test_file_path)
        
        assert isinstance(pages, list)
        if loader.can_handle_paginate(test_file_path):
            assert len(pages) > 0
            for page in pages:
                assert "content" in page
                assert isinstance(page["content"], str)

    def test_no_ocr_loading(self, loader_no_ocr, test_file_path):
        """Test loading with OCR disabled"""
        pages = loader_no_ocr.load(test_file_path)
        
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert "content" in pages[0]

    def test_config_features(self, test_file_path):
        """Test various configuration features"""
        # Test with custom OCR settings
        config = DoclingConfig(
            ocr_enabled=True,
            tesseract_cmd="/opt/homebrew/bin/tesseract",
            force_full_page_ocr=True
        )
        loader = DocumentLoaderDocling(config)
        pages = loader.load(test_file_path)
        assert len(pages) > 0

        # Test with custom table settings
        config = DoclingConfig(
            table_structure_enabled=True,
            do_cell_matching=False
        )
        loader = DocumentLoaderDocling(config)
        pages = loader.load(test_file_path)
        assert len(pages) > 0

    def test_simple_initialization(self, test_file_path):
        """Test simple initialization and basic functionality without any special configurations"""
        # Simple initialization like before
        loader = DocumentLoaderDocling()
        
        # Basic load and verify
        pages = loader.load(test_file_path)
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert "content" in pages[0]
        assert isinstance(pages[0]["content"], str)
        assert len(pages[0]["content"]) > 0  # Should have extracted some text

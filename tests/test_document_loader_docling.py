import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from extract_thinker.document_loader.document_loader_docling import DocumentLoaderDocling
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
    def loader(self, default_pipeline_options):
        format_options = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=default_pipeline_options)
        }
        return DocumentLoaderDocling(format_options=format_options)

    @pytest.fixture
    def loader_no_ocr(self):
        """Loader instance with OCR disabled"""
        pipeline_options = PdfPipelineOptions(
            do_table_structure=True,
            do_ocr=False
        )
        format_options = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
        return DocumentLoaderDocling(format_options=format_options)

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

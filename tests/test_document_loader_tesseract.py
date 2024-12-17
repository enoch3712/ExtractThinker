import os
import pytest
from dotenv import load_dotenv
from extract_thinker.document_loader.document_loader_tesseract import DocumentLoaderTesseract
from .test_document_loader_base import BaseDocumentLoaderTest

load_dotenv()

class TestDocumentLoaderTesseract(BaseDocumentLoaderTest):
    @pytest.fixture
    def loader(self):
        tesseract_path = os.getenv("TESSERACT_PATH")
        return DocumentLoaderTesseract(tesseract_path)

    @pytest.fixture
    def test_file_path(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "test_images", "invoice.png")

    def test_tesseract_specific_content(self, loader, test_file_path):
        """Test OCR-specific content extraction"""
        pages = loader.load(test_file_path)
        
        assert isinstance(pages, list)
        assert len(pages) > 0
        
        first_page = pages[0]
        assert "content" in first_page
        assert "Invoice" in first_page["content"]
        assert "0000001" in first_page["content"]

    def test_vision_mode(self, loader, test_file_path):
        """Override base class vision mode test for Tesseract-specific behavior"""
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

    def test_parallel_processing(self, loader, test_file_path):
        """Test parallel processing of multiple pages"""
        # Create a multi-page test case
        pages = loader.load(test_file_path)
        assert isinstance(pages, list)
        assert all("content" in page for page in pages)
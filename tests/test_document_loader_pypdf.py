import os
import pytest
from extract_thinker.document_loader.document_loader_pypdf import DocumentLoaderPyPdf
from .test_document_loader_base import BaseDocumentLoaderTest

class TestDocumentLoaderPyPdf(BaseDocumentLoaderTest):
    @pytest.fixture
    def loader(self):
        return DocumentLoaderPyPdf()
        
    @pytest.fixture
    def test_file_path(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, 'files', 'CV_Candidate.pdf')
        
    def test_pdf_specific_content(self, loader, test_file_path):
        """Test PDF-specific content extraction"""
        pages = loader.load(test_file_path)
        
        # Verify basic structure
        assert isinstance(pages, list)
        assert len(pages) > 0
        
        # Check first page content
        first_page = pages[0]
        assert isinstance(first_page, dict)
        assert "content" in first_page
        
        # Verify expected content is present
        content = first_page["content"]
        assert "Universityof NewYork" in content
        assert "XYZInnovations" in content

    def test_vision_mode(self, loader, test_file_path):
        """Test vision mode functionality"""
        # Enable vision mode
        loader.set_vision_mode(True)
        
        # Load document
        pages = loader.load(test_file_path)
        
        # Verify basic structure
        assert isinstance(pages, list)
        assert len(pages) > 0
        
        # Check first page
        first_page = pages[0]
        assert isinstance(first_page, dict)
        assert "content" in first_page
        assert "image" in first_page
        
        # Verify image data
        assert isinstance(first_page["image"], bytes)

    def test_cache_functionality(self, loader, test_file_path):
        """Test that caching works correctly"""
        # First load
        result1 = loader.load(test_file_path)
        
        # Second load should come from cache
        result2 = loader.load(test_file_path)
        
        # Results should be identical
        assert result1 == result2
        
        # Content should be the same
        assert result1[0]["content"] == result2[0]["content"]

    def test_invalid_file(self, loader):
        """Test handling of invalid files"""
        with pytest.raises(ValueError):
            loader.load("nonexistent.pdf")
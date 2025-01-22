import os
import pytest
from extract_thinker.document_loader.document_loader_pypdf import DocumentLoaderPyPdf, PyPDFConfig
from .test_document_loader_base import BaseDocumentLoaderTest

class TestDocumentLoaderPyPdf(BaseDocumentLoaderTest):
    @pytest.fixture
    def pdf_config(self):
        """Default PyPDF configuration for testing"""
        return PyPDFConfig(
            vision_enabled=True,
            extract_text=True
        )

    @pytest.fixture
    def loader(self, pdf_config):
        return DocumentLoaderPyPdf(pdf_config)
        
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

    def test_config_validation(self):
        """Test configuration validation"""
        # Test invalid cache_ttl
        with pytest.raises(ValueError, match="cache_ttl must be positive"):
            PyPDFConfig(cache_ttl=0)

        # Test invalid password type
        with pytest.raises(ValueError, match="password must be a string"):
            PyPDFConfig(password=123)

    def test_text_extraction_control(self, test_file_path):
        """Test control over text extraction"""
        # Test with text extraction disabled
        config = PyPDFConfig(extract_text=False)
        loader = DocumentLoaderPyPdf(config)
        pages = loader.load(test_file_path)
        assert pages[0]["content"] == ""

        # Test with text extraction enabled (default)
        config = PyPDFConfig(extract_text=True)
        loader = DocumentLoaderPyPdf(config)
        pages = loader.load(test_file_path)
        assert pages[0]["content"] != ""

    def test_vision_mode(self, loader, test_file_path):
        """Test vision mode functionality"""
        # Vision mode should be enabled via config
        assert loader.vision_mode is True
        assert loader.can_handle_vision(test_file_path) is True
        
        # Test loading with vision mode
        pages = loader.load(test_file_path)
        assert len(pages) > 0
        assert "image" in pages[0]
        assert isinstance(pages[0]["image"], bytes)
        
        # Test disabling vision mode
        loader.set_vision_mode(False)
        assert loader.vision_mode is False
        assert loader.can_handle_vision(test_file_path) is False
        pages = loader.load(test_file_path)
        assert "image" not in pages[0]

    def test_simple_initialization(self, test_file_path):
        """Test simple initialization without configuration"""
        # Basic initialization
        loader = DocumentLoaderPyPdf()
        
        # Load and verify basic functionality
        pages = loader.load(test_file_path)
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert "content" in pages[0]
        assert pages[0]["content"] != ""
        
        # Vision mode should be off by default
        assert loader.vision_mode is False
        assert "image" not in pages[0]

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
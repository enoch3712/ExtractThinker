import os
import pytest
from extract_thinker.document_loader.document_loader_beautiful_soup import DocumentLoaderBeautifulSoup, BeautifulSoupConfig
from .test_document_loader_base import BaseDocumentLoaderTest

class TestDocumentLoaderBeautifulSoup(BaseDocumentLoaderTest):
    @pytest.fixture
    def bs_config(self):
        return BeautifulSoupConfig(
            header_handling="summarize",
            max_tokens=1000,
            request_timeout=10
        )

    @pytest.fixture
    def loader(self, bs_config):
        return DocumentLoaderBeautifulSoup(bs_config)

    @pytest.fixture
    def test_file_path(self):
        return "https://www.google.com"  # Using a stable website for testing

    def test_beautifulsoup_specific_content(self, loader, test_file_path):
        """Test web content extraction"""
        pages = loader.load(test_file_path)
        
        assert isinstance(pages, list)
        assert len(pages) > 0
        
        first_page = pages[0]
        assert "content" in first_page
        assert "Google" in first_page["content"]

    def test_header_handling(self, loader, test_file_path):
        """Test header handling functionality"""
        pages = loader.load(test_file_path)
        first_page = pages[0]
        assert "content" in first_page
        # Headers should be processed according to the header_handling setting
        assert isinstance(first_page["content"], str)

    def test_invalid_url(self, loader):
        """Test handling of invalid URLs"""
        with pytest.raises(ValueError):
            loader.load("https://this-is-an-invalid-url-that-should-not-exist.com")

    def test_vision_mode(self, loader, test_file_path):
        """Test that vision mode is not supported for BeautifulSoup loader"""
        loader.set_vision_mode(True)
        with pytest.raises(ValueError):
            loader.load(test_file_path)

    def test_config_validation(self):
        """Test BeautifulSoupConfig validation"""
        # Test invalid header handling
        with pytest.raises(ValueError, match="Invalid header_handling value"):
            BeautifulSoupConfig(header_handling="invalid")

        # Test invalid max_tokens
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            BeautifulSoupConfig(max_tokens=0)

        # Test invalid request_timeout
        with pytest.raises(ValueError, match="request_timeout must be positive"):
            BeautifulSoupConfig(request_timeout=0)

    def test_custom_elements_removal(self, bs_config, test_file_path):
        """Test custom elements removal"""
        config = BeautifulSoupConfig(
            header_handling="summarize",
            remove_elements=['div', 'span']  # Custom elements to remove
        )
        loader = DocumentLoaderBeautifulSoup(config)
        pages = loader.load(test_file_path)
        assert len(pages) > 0

    def test_simple_initialization(self):
        """Test simple initialization and basic functionality without any special configurations"""
        # Simple initialization like before
        loader = DocumentLoaderBeautifulSoup()
        
        # Test with Google's homepage
        pages = loader.load("https://www.google.com")
        
        # Basic validations
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert isinstance(pages[0], dict)
        assert "content" in pages[0]
        assert isinstance(pages[0]["content"], str)
        assert len(pages[0]["content"]) > 0  # Should have extracted some text
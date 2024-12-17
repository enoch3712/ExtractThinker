import os
import pytest
from extract_thinker.document_loader.beautiful_soup_web_loader import DocumentLoaderBeautifulSoup
from .test_document_loader_base import BaseDocumentLoaderTest

class TestDocumentLoaderBeautifulSoup(BaseDocumentLoaderTest):
    @pytest.fixture
    def loader(self):
        return DocumentLoaderBeautifulSoup(header_handling="summarize")

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
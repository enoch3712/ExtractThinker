import os
import pytest
from extract_thinker.document_loader.document_loader_doc2txt import DocumentLoaderDoc2txt, Doc2txtConfig
from tests.test_document_loader_base import BaseDocumentLoaderTest


class TestDocumentLoaderDoc2txt(BaseDocumentLoaderTest):
    @pytest.fixture
    def doc_config(self):
        """Default Doc2txt configuration for testing"""
        return Doc2txtConfig(
            preserve_whitespace=False,
            page_separator='\n\n\n'
        )

    @pytest.fixture
    def loader(self, doc_config):
        return DocumentLoaderDoc2txt(doc_config)

    @pytest.fixture
    def test_file_path(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, 'files', 'invoice.docx')

    def test_word_specific_content(self, loader, test_file_path):
        """Test Word document-specific content extraction"""
        pages = loader.load(test_file_path)
        
        assert isinstance(pages, list)
        assert len(pages) > 0
        
        # Word documents are split into paragraphs as pages
        first_page = pages[0]
        assert "content" in first_page
        assert len(first_page["content"]) > 0

    def test_vision_mode(self, loader):
        """Test that vision mode is not supported for Word loader"""
        # Vision mode should be disabled by default
        assert loader.vision_mode is False
        
        # Attempting to enable vision mode should raise an error
        with pytest.raises(ValueError, match="Vision mode is not supported"):
            loader.set_vision_mode(True)
        
        # Vision mode should still be False after failed attempt
        assert loader.vision_mode is False
        
        # can_handle_vision should always return False
        assert loader.can_handle_vision("test.docx") is False

    def test_multiple_pages(self, loader, test_file_path):
        """Test that Word documents with multiple paragraphs are split into multiple pages"""
        pages = loader.load(test_file_path)
        
        assert isinstance(pages, list)
        assert len(pages) > 1, "Document should contain multiple pages/paragraphs"
        
        # Verify each page has content
        for page in pages:
            assert "content" in page
            assert len(page["content"]) > 0

    def test_config_validation(self):
        """Test configuration validation"""
        # Test invalid cache_ttl
        with pytest.raises(ValueError, match="cache_ttl must be positive"):
            Doc2txtConfig(cache_ttl=0)

        # Test invalid page_separator
        with pytest.raises(ValueError, match="page_separator must be a string"):
            Doc2txtConfig(page_separator=123)

        # Test unsupported image extraction
        with pytest.raises(ValueError, match="Image extraction is not supported"):
            Doc2txtConfig(extract_images=True)

    def test_simple_initialization(self, test_file_path):
        """Test simple initialization without configuration"""
        # Basic initialization
        loader = DocumentLoaderDoc2txt()
        
        # Load and verify basic functionality
        pages = loader.load(test_file_path)
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert "content" in pages[0]
        assert len(pages[0]["content"]) > 0
        
        # Vision mode should be off by default
        assert loader.vision_mode is False
import os
import pytest
from extract_thinker.document_loader.document_loader_txt import DocumentLoaderTxt, TxtConfig
from tests.test_document_loader_base import BaseDocumentLoaderTest


class TestDocumentLoaderTxt(BaseDocumentLoaderTest):
    @pytest.fixture
    def txt_config(self):
        """Default TXT configuration for testing"""
        return TxtConfig(
            encoding='utf-8',
            preserve_whitespace=False,
            split_paragraphs=False
        )

    @pytest.fixture
    def loader(self, txt_config):
        return DocumentLoaderTxt(txt_config)

    @pytest.fixture
    def test_file_path(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, 'files', 'ambiguous_credit_note.txt')

    def test_txt_specific_content(self, loader, test_file_path):
        """Test text file-specific content extraction"""
        pages = loader.load(test_file_path)
        
        assert isinstance(pages, list)
        assert len(pages) > 0
        
        first_page = pages[0]
        assert "content" in first_page
        assert "CREDIT NOTE / RECEIPT" in first_page["content"]
        assert "Customer: John Smith" in first_page["content"]
        assert "Payment Method: Store Credit" in first_page["content"]

    def test_config_validation(self):
        """Test configuration validation"""
        # Test invalid cache_ttl
        with pytest.raises(ValueError, match="cache_ttl must be positive"):
            TxtConfig(cache_ttl=0)

        # Test invalid encoding type
        with pytest.raises(ValueError, match="encoding must be a string"):
            TxtConfig(encoding=123)

    def test_whitespace_handling(self, test_file_path):
        """Test whitespace preservation settings"""
        # Test with whitespace preservation
        config = TxtConfig(preserve_whitespace=True)
        loader = DocumentLoaderTxt(config)
        pages = loader.load(test_file_path)
        assert len(pages) > 0
        assert pages[0]["content"].count('\n') >= pages[0]["content"].strip().count('\n')

        # Test without whitespace preservation (default)
        config = TxtConfig(preserve_whitespace=False)
        loader = DocumentLoaderTxt(config)
        pages = loader.load(test_file_path)
        assert len(pages) > 0
        assert pages[0]["content"] == pages[0]["content"].strip()

    def test_paragraph_splitting(self, test_file_path):
        """Test paragraph splitting functionality"""
        # Test with paragraph splitting enabled
        config = TxtConfig(split_paragraphs=True)
        loader = DocumentLoaderTxt(config)
        pages = loader.load(test_file_path)
        assert len(pages) > 1, "Should split content into multiple pages"

        # Test without paragraph splitting (default)
        config = TxtConfig(split_paragraphs=False)
        loader = DocumentLoaderTxt(config)
        pages = loader.load(test_file_path)
        assert len(pages) == 1, "Should keep content as single page"

    def test_vision_mode(self, loader):
        """Test that vision mode is not supported"""
        # Vision mode should be disabled by default
        assert loader.vision_mode is False
        
        # Attempting to enable vision mode should raise an error
        with pytest.raises(ValueError, match="Vision mode is not supported"):
            loader.set_vision_mode(True)
        
        # Vision mode should still be False after failed attempt
        assert loader.vision_mode is False
        
        # can_handle_vision should always return False
        assert loader.can_handle_vision("test.txt") is False

    def test_simple_initialization(self, test_file_path):
        """Test simple initialization without configuration"""
        # Basic initialization
        loader = DocumentLoaderTxt()
        
        # Load and verify basic functionality
        pages = loader.load(test_file_path)
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert "content" in pages[0]
        assert "CREDIT NOTE / RECEIPT" in pages[0]["content"]
        
        # Vision mode should be off by default
        assert loader.vision_mode is False

    def test_encoding_handling(self, tmp_path):
        """Test different text encodings"""
        # Create a test file with non-ASCII content
        test_content = "Hello 世界"  # Mixed ASCII and Unicode
        test_file = tmp_path / "test_encoding.txt"
        
        # Test UTF-8
        test_file.write_text(test_content, encoding='utf-8')
        config = TxtConfig(encoding='utf-8')
        loader = DocumentLoaderTxt(config)
        pages = loader.load(str(test_file))
        assert pages[0]["content"] == test_content
import os
import pytest
from extract_thinker.document_loader.document_loader_markitdown import DocumentLoaderMarkItDown, MarkItDownConfig
from tests.test_document_loader_base import BaseDocumentLoaderTest
from io import BytesIO

class TestDocumentLoaderMarkItDown(BaseDocumentLoaderTest):
    @pytest.fixture
    def markitdown_config(self):
        """Default MarkItDown configuration for testing"""
        return MarkItDownConfig(
            mime_type_detection=True,
            preserve_whitespace=False,
            default_extension='txt'
        )

    @pytest.fixture
    def loader(self, markitdown_config):
        return DocumentLoaderMarkItDown(markitdown_config)

    @pytest.fixture
    def test_file_path(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, 'files', 'invoice.pdf')

    def test_markitdown_specific_content(self, loader, test_file_path):
        """Test MarkItDown-specific content extraction"""
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

    def test_config_validation(self):
        """Test configuration validation"""
        # Test invalid mime_type_detection
        with pytest.raises(ValueError, match="mime_type_detection must be a boolean"):
            MarkItDownConfig(mime_type_detection="yes")

        # Test empty default_extension
        with pytest.raises(ValueError, match="default_extension cannot be empty"):
            MarkItDownConfig(default_extension="")

        # Test empty page_separator
        with pytest.raises(ValueError, match="page_separator cannot be empty"):
            MarkItDownConfig(page_separator="")

    def test_whitespace_handling(self, markitdown_config, test_file_path):
        """Test whitespace handling configurations"""
        # Test with whitespace preservation
        config = MarkItDownConfig(
            preserve_whitespace=True
        )
        loader = DocumentLoaderMarkItDown(config)
        pages = loader.load(test_file_path)
        assert len(pages) > 0

        # Test without whitespace preservation
        config = MarkItDownConfig(
            preserve_whitespace=False
        )
        loader = DocumentLoaderMarkItDown(config)
        pages = loader.load(test_file_path)
        assert len(pages) > 0

    def test_mime_type_detection(self, markitdown_config, test_file_path):
        """Test MIME type detection configurations"""
        # Test with MIME type detection
        config = MarkItDownConfig(
            mime_type_detection=True,
            default_extension='txt'
        )
        loader = DocumentLoaderMarkItDown(config)
        pages = loader.load(test_file_path)
        assert len(pages) > 0

        # Test without MIME type detection
        config = MarkItDownConfig(
            mime_type_detection=False,
            default_extension='pdf'
        )
        loader = DocumentLoaderMarkItDown(config)
        pages = loader.load(test_file_path)
        assert len(pages) > 0

    def test_simple_initialization(self, test_file_path):
        """Test simple initialization and basic functionality without any special configurations"""
        # Simple initialization like before
        loader = DocumentLoaderMarkItDown()
        
        # Basic load and verify
        pages = loader.load(test_file_path)
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert "content" in pages[0]
        assert isinstance(pages[0]["content"], str)
        assert len(pages[0]["content"]) > 0  # Should have extracted some text
        
        # Test with a BytesIO stream
        with open(test_file_path, 'rb') as f:
            stream = BytesIO(f.read())
            pages = loader.load(stream)
            assert len(pages) > 0
            assert isinstance(pages[0]["content"], str)
            
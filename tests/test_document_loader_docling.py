import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from extract_thinker.document_loader.document_loader_docling import DocumentLoaderDocling
from tests.test_document_loader_base import BaseDocumentLoaderTest
from io import BytesIO

class TestDocumentLoaderDocling(BaseDocumentLoaderTest):
    @pytest.fixture
    def loader(self):
        return DocumentLoaderDocling()

    @pytest.fixture
    def test_file_path(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, 'files', 'document.pdf')

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
                

if __name__ == "__main__":
    loader = DocumentLoaderDocling()
    loader.set_vision_mode(True)
    current_dir = os.getcwd()
    test_file_path = os.path.join(current_dir, 'tests', 'files', 'Regional_GDP_per_capita_2018_2.pdf')
    content = loader.load(test_file_path)
    print(content)
import os
import pytest
from extract_thinker.document_loader.document_loader_txt import DocumentLoaderTxt
from .test_document_loader_base import BaseDocumentLoaderTest

class TestDocumentLoaderTxt(BaseDocumentLoaderTest):
    @pytest.fixture
    def loader(self):
        return DocumentLoaderTxt()

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
        assert "CN-2024-001" in first_page["content"]
        assert "John Smith" in first_page["content"] 
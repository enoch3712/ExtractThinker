import os
import pytest
from extract_thinker.document_loader.document_loader_doc2txt import DocumentLoaderDoc2txt
from tests.test_document_loader_base import BaseDocumentLoaderTest


class TestDocumentLoaderDoc2txt(BaseDocumentLoaderTest):
    @pytest.fixture
    def loader(self):
        return DocumentLoaderDoc2txt()

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

    def test_vision_mode(self, loader, test_file_path):
        """Test that vision mode is not supported for BeautifulSoup loader"""
        loader.set_vision_mode(True)
        with pytest.raises(ValueError):
            loader.load(test_file_path)

    def test_multiple_pages(self, loader, test_file_path):
        """Test that Word documents with multiple paragraphs are split into multiple pages"""
        pages = loader.load(test_file_path)
        
        assert isinstance(pages, list)
        assert len(pages) > 1, "Document should contain multiple pages/paragraphs"
        
        # Verify each page has content
        for page in pages:
            assert "content" in page
            assert len(page["content"]) > 0
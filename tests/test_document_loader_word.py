import os
import pytest
from extract_thinker.document_loader.document_loader_doc2txt import DocumentLoaderDoc2txt
from .test_document_loader_base import BaseDocumentLoaderTest

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

    def test_vision_mode_not_supported(self, loader, test_file_path):
        """Test that vision mode is not supported for Word documents"""
        loader.set_vision_mode(True)
        with pytest.raises(ValueError, match="Source cannot be processed in vision mode"):
            loader.load(test_file_path)
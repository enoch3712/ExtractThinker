import os
from pytest import fixture
from extract_thinker.document_loader.document_loader_pdfplumber import DocumentLoaderPdfPlumber
from extract_thinker.extractor import Extractor
from .test_document_loader_base import BaseDocumentLoaderTest
from tests.models.invoice import InvoiceContract
from dotenv import load_dotenv

load_dotenv()
cwd = os.getcwd()

class TestDocumentLoaderPdfPlumber(BaseDocumentLoaderTest):
    @fixture
    def loader(self):
        return DocumentLoaderPdfPlumber()

    @fixture
    def test_file_path(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, 'files', 'invoice.pdf')

    def test_pdfplumber_specific_content(self, loader, test_file_path):
        """Test PDF-specific content extraction"""
        pages = loader.load(test_file_path)
        
        assert isinstance(pages, list)
        assert len(pages) > 0
        
        first_page = pages[0]
        assert "content" in first_page
        assert "tables" in first_page
        assert len(first_page["tables"]) > 0

    def test_vision_mode(self, loader, test_file_path):
        """Override base class vision mode test for PDF-specific behavior"""
        loader.set_vision_mode(True)
        pages = loader.load(test_file_path)
        
        assert isinstance(pages, list)
        assert len(pages) > 0
        
        for page in pages:
            assert isinstance(page, dict)
            assert "content" in page
            assert "tables" in page
            if loader.can_handle_vision(test_file_path):
                assert "image" in page
                assert isinstance(page["image"], bytes)
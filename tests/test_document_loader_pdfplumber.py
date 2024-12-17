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

def test_basic_extractor_functionality():
    # Arrange
    test_file_path = os.path.join(cwd, "tests", "files", "invoice.pdf")
    
    extractor = Extractor()
    extractor.load_document_loader(DocumentLoaderPdfPlumber())
    extractor.load_llm("gpt-4o-mini")
    
    # Act
    result = extractor.extract(test_file_path, InvoiceContract)
    
    # Assert
    assert result is not None
    assert isinstance(result, InvoiceContract)
    assert hasattr(result, 'invoice_number')
    assert hasattr(result, 'invoice_date')
    assert hasattr(result, 'lines')
    assert len(result.lines) > 0
    
    # Check specific invoice details
    assert result.invoice_number == "00012"
    assert result.invoice_date == "1/30/23"
    assert result.total_amount == 1125
    
    # Check line items
    first_line = result.lines[0]
    assert first_line.description == "Consultation services"
    assert first_line.quantity == 3
    assert first_line.unit_price == 375
    assert first_line.amount == 1125


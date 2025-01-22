import os
from pytest import fixture
import pytest
from extract_thinker.document_loader.document_loader_pdfplumber import DocumentLoaderPdfPlumber, PDFPlumberConfig
from extract_thinker.extractor import Extractor
from .test_document_loader_base import BaseDocumentLoaderTest
from tests.models.invoice import InvoiceContract
from dotenv import load_dotenv

load_dotenv()
cwd = os.getcwd()

class TestDocumentLoaderPdfPlumber(BaseDocumentLoaderTest):
    @fixture
    def pdf_config(self):
        """Default PDFPlumber configuration for testing"""
        return PDFPlumberConfig(
            vision_enabled=True,
            extract_tables=True,
            table_settings={
                'vertical_strategy': 'text',
                'horizontal_strategy': 'text',
                'intersection_y_tolerance': 10
            }
        )

    @fixture
    def loader(self, pdf_config):
        return DocumentLoaderPdfPlumber(pdf_config)

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

    def test_config_validation(self):
        """Test configuration validation"""
        # Test invalid cache_ttl
        with pytest.raises(ValueError, match="cache_ttl must be positive"):
            PDFPlumberConfig(cache_ttl=0)

        # Test invalid table_settings
        with pytest.raises(ValueError, match="table_settings must be a dictionary"):
            PDFPlumberConfig(table_settings="invalid")

    def test_table_extraction_control(self, test_file_path):
        """Test control over table extraction"""
        # Test with tables disabled
        config = PDFPlumberConfig(extract_tables=False)
        loader = DocumentLoaderPdfPlumber(config)
        pages = loader.load(test_file_path)
        assert len(pages[0]["tables"]) == 0

        # Test with custom table settings
        custom_settings = {
            'vertical_strategy': 'lines',
            'horizontal_strategy': 'lines'
        }
        config = PDFPlumberConfig(table_settings=custom_settings)
        loader = DocumentLoaderPdfPlumber(config)
        pages = loader.load(test_file_path)
        assert isinstance(pages[0]["tables"], list)

    def test_vision_mode(self, loader, test_file_path):
        """Test vision mode functionality"""
        # Vision mode should be enabled via config
        assert loader.vision_mode is True
        assert loader.can_handle_vision(test_file_path) is True
        
        # Test loading with vision mode
        pages = loader.load(test_file_path)
        assert len(pages) > 0
        assert "image" in pages[0]
        assert isinstance(pages[0]["image"], bytes)
        
        # Test disabling vision mode
        loader.set_vision_mode(False)
        assert loader.vision_mode is False
        assert loader.can_handle_vision(test_file_path) is False
        pages = loader.load(test_file_path)
        assert "image" not in pages[0]

    def test_simple_initialization(self, test_file_path):
        """Test simple initialization without configuration"""
        # Basic initialization
        loader = DocumentLoaderPdfPlumber()
        
        # Load and verify basic functionality
        pages = loader.load(test_file_path)
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert "content" in pages[0]
        assert "tables" in pages[0]
        
        # Vision mode should be off by default
        assert loader.vision_mode is False
        assert "image" not in pages[0]
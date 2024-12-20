import os
import pytest
from dotenv import load_dotenv
from extract_thinker.document_loader.document_loader_azure_document_intelligence import DocumentLoaderAzureForm
from tests.test_document_loader_base import BaseDocumentLoaderTest

load_dotenv()

class TestDocumentLoaderAzureForm(BaseDocumentLoaderTest):
    @pytest.fixture
    def loader(self):
        return DocumentLoaderAzureForm(
            subscription_key=os.getenv("AZURE_SUBSCRIPTION_KEY"),
            endpoint=os.getenv("AZURE_ENDPOINT")
        )

    @pytest.fixture
    def test_file_path(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, 'test_images', 'invoice.png')
    
    def test_vision_mode(self, loader, test_file_path):
        """Override base class vision mode test for Azure Document Intelligence-specific behavior"""
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

    def test_azure_specific_content(self, loader, test_file_path):
        """Test Azure-specific content extraction"""
        pages = loader.load(test_file_path)
        
        assert isinstance(pages, list)
        assert len(pages) > 0
        
        first_page = pages[0]
        assert "content" in first_page
        assert "tables" in first_page
        assert "Invoice 0000001" in first_page["content"]
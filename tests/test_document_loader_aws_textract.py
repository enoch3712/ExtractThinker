import os
import pytest
from dotenv import load_dotenv
from extract_thinker.document_loader.document_loader_aws_textract import DocumentLoaderAWSTextract
from tests.test_document_loader_base import BaseDocumentLoaderTest

load_dotenv()

class TestDocumentLoaderAWSTextract(BaseDocumentLoaderTest):
    @pytest.fixture
    def loader(self):
        return DocumentLoaderAWSTextract(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION_NAME")
        )

    @pytest.fixture
    def test_file_path(self):
        current_dir = os.getcwd()
        return os.path.join(current_dir, 'tests', 'files', 'invoice.pdf')

    def test_textract_specific_content(self, loader, test_file_path):
        """Test Textract-specific content extraction"""
        pages = loader.load(test_file_path)
        
        assert isinstance(pages, list)
        assert len(pages) > 0
        
        first_page = pages[0]
        assert "content" in first_page
        assert "tables" in first_page
    
    def test_vision_mode(self, loader, test_file_path):
        """Override base class vision mode test for AWS Textract-specific behavior"""
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
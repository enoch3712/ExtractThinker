import os
import pytest
import warnings
from dotenv import load_dotenv
from extract_thinker.document_loader.document_loader_google_document_ai import (
    DocumentLoaderGoogleDocumentAI,
    DocumentLoaderDocumentAI
)
from tests.test_document_loader_base import BaseDocumentLoaderTest

load_dotenv()

class TestDocumentLoaderGoogleDocumentAI(BaseDocumentLoaderTest):
    @pytest.fixture
    def loader(self):
        return DocumentLoaderGoogleDocumentAI(
            project_id=os.getenv("DOCUMENTAI_PROJECT_ID"),
            location=os.getenv("DOCUMENTAI_LOCATION"),
            processor_id=os.getenv("DOCUMENTAI_PROCESSOR_ID"),
            credentials=os.getenv("DOCUMENTAI_GOOGLE_CREDENTIALS")
        )

    def test_deprecation_warning(self):
        """Test that using old class name raises deprecation warning"""
        with pytest.warns(DeprecationWarning) as record:
            DocumentLoaderDocumentAI(
                project_id=os.getenv("DOCUMENTAI_PROJECT_ID"),
                location=os.getenv("DOCUMENTAI_LOCATION"),
                processor_id=os.getenv("DOCUMENTAI_PROCESSOR_ID"),
                credentials=os.getenv("DOCUMENTAI_GOOGLE_CREDENTIALS")
            )
        
        # Verify the warning message
        assert len(record) == 1
        assert "DocumentLoaderDocumentAI is deprecated" in str(record[0].message)
        assert "Use DocumentLoaderGoogleDocumentAI instead" in str(record[0].message)

    @pytest.fixture
    def test_file_path(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, 'files', 'form_with_tables.pdf')

    def test_documentai_specific_content(self, loader, test_file_path):
        """Test Document AI-specific content extraction"""
        pages = loader.load(test_file_path)
        
        assert isinstance(pages, list)
        assert len(pages) > 0
        
        first_page = pages[0]
        assert "content" in first_page
        assert "tables" in first_page

    def test_vision_mode(self, loader, test_file_path):
        """Override base class vision mode test for Document AI-specific behavior"""
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
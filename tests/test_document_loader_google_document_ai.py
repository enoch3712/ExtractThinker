import os
import pytest
from dotenv import load_dotenv
from extract_thinker.document_loader.document_loader_google_document_ai import DocumentLoaderDocumentAI
from .test_document_loader_base import BaseDocumentLoaderTest

load_dotenv()

class TestDocumentLoaderGoogleDocumentAI(BaseDocumentLoaderTest):
    @pytest.fixture
    def loader(self):
        return DocumentLoaderDocumentAI(
            credentials=os.getenv("DOCUMENTAI_GOOGLE_CREDENTIALS"),
            location=os.getenv("DOCUMENTAI_LOCATION"),
            processor_name=os.getenv("DOCUMENTAI_PROCESSOR_NAME")
        )

    @pytest.fixture
    def test_file_path(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, 'files', 'CV_Candidate.pdf')

    def test_documentai_specific_content(self, loader, test_file_path):
        """Test Document AI-specific content extraction"""
        pages = loader.load(test_file_path)
        
        assert isinstance(pages, list)
        assert len(pages) > 0
        
        first_page = pages[0]
        assert "content" in first_page
        assert "paragraphs" in first_page
        assert "tables" in first_page
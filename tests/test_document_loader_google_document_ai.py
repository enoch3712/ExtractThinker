import os
import pytest
import warnings
from dotenv import load_dotenv
from extract_thinker.document_loader.document_loader_google_document_ai import (
    DocumentLoaderGoogleDocumentAI,
    DocumentLoaderDocumentAI,
    GoogleDocAIConfig
)
from tests.test_document_loader_base import BaseDocumentLoaderTest

load_dotenv()

class TestDocumentLoaderGoogleDocumentAI(BaseDocumentLoaderTest):
    @pytest.fixture
    def config(self):
        """Base config fixture with Google Document AI credentials."""
        return GoogleDocAIConfig(
            project_id=os.getenv("DOCUMENTAI_PROJECT_ID"),
            location=os.getenv("DOCUMENTAI_LOCATION"),
            processor_id=os.getenv("DOCUMENTAI_PROCESSOR_ID"),
            credentials=os.getenv("DOCUMENTAI_GOOGLE_CREDENTIALS")
        )

    @pytest.fixture
    def loader(self, config):
        """Default loader with base config."""
        return DocumentLoaderGoogleDocumentAI(config)

    @pytest.fixture
    def test_file_path(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, 'files', 'form_with_tables.pdf')

    def test_config_validation(self):
        """Test GoogleDocAIConfig validation."""
        # Test valid config
        valid_config = GoogleDocAIConfig(
            project_id="test-project",
            location="us",
            processor_id="test-processor",
            credentials="{}"  # Minimal valid JSON
        )
        assert valid_config.processor_version == "rc"  # Default value
        
        # Test invalid page range
        with pytest.raises(ValueError) as exc_info:
            GoogleDocAIConfig(
                project_id="test-project",
                location="us",
                processor_id="test-processor",
                credentials="{}",
                page_range=[-1, 0]  # Invalid page numbers
            )
        assert "page_range must be a list of positive integers" in str(exc_info.value)
        
        # Test empty required fields
        for field in ["project_id", "location", "processor_id", "credentials"]:
            with pytest.raises(ValueError) as exc_info:
                config_dict = {
                    "project_id": "test-project",
                    "location": "us",
                    "processor_id": "test-processor",
                    "credentials": "{}"
                }
                config_dict[field] = ""
                GoogleDocAIConfig(**config_dict)
            assert f"{field} cannot be empty" in str(exc_info.value)

    def test_legacy_initialization(self, test_file_path):
        """Test old-style (legacy) initialization and functionality."""
        # Test basic initialization
        loader = DocumentLoaderGoogleDocumentAI(
            project_id=os.getenv("DOCUMENTAI_PROJECT_ID"),
            location=os.getenv("DOCUMENTAI_LOCATION"),
            processor_id=os.getenv("DOCUMENTAI_PROCESSOR_ID"),
            credentials=os.getenv("DOCUMENTAI_GOOGLE_CREDENTIALS")
        )
        assert isinstance(loader.config, GoogleDocAIConfig)
        assert loader.config.processor_version == "rc"  # Default value
        
        # Test full initialization with custom parameters
        loader_full = DocumentLoaderGoogleDocumentAI(
            project_id=os.getenv("DOCUMENTAI_PROJECT_ID"),
            location=os.getenv("DOCUMENTAI_LOCATION"),
            processor_id=os.getenv("DOCUMENTAI_PROCESSOR_ID"),
            credentials=os.getenv("DOCUMENTAI_GOOGLE_CREDENTIALS"),
            content="test_content",
            cache_ttl=600,
            enable_native_pdf_parsing=True,
            page_range=[1, 2, 3]
        )
        assert isinstance(loader_full.config, GoogleDocAIConfig)
        assert loader_full.config.content == "test_content"
        assert loader_full.config.cache_ttl == 600
        assert loader_full.config.enable_native_pdf_parsing is True
        assert loader_full.config.page_range == [1, 2, 3]
        
        # Test actual document processing
        pages = loader_full.load(test_file_path)
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert "content" in pages[0]
        assert "tables" in pages[0]
        assert isinstance(pages[0]["content"], str)
        assert isinstance(pages[0]["tables"], list)

    def test_deprecation_warning(self):
        """Test that using old class name raises deprecation warning."""
        with pytest.warns(DeprecationWarning) as record:
            DocumentLoaderDocumentAI(
                project_id=os.getenv("DOCUMENTAI_PROJECT_ID"),
                location=os.getenv("DOCUMENTAI_LOCATION"),
                processor_id=os.getenv("DOCUMENTAI_PROCESSOR_ID"),
                credentials=os.getenv("DOCUMENTAI_GOOGLE_CREDENTIALS")
            )
        
        assert len(record) == 1
        assert "DocumentLoaderDocumentAI is deprecated" in str(record[0].message)
        assert "Use DocumentLoaderGoogleDocumentAI instead" in str(record[0].message)

    def test_vision_mode(self, loader, test_file_path):
        """Test vision mode with config-based loader."""
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
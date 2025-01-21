import os
import pytest
from dotenv import load_dotenv
from extract_thinker.document_loader.document_loader_aws_textract import (
    DocumentLoaderAWSTextract,
    TextractConfig
)
from tests.test_document_loader_base import BaseDocumentLoaderTest

load_dotenv()

class TestDocumentLoaderAWSTextract(BaseDocumentLoaderTest):
    @pytest.fixture
    def config(self):
        """Base config fixture with AWS credentials."""
        return TextractConfig(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION_NAME")
        )

    @pytest.fixture
    def loader(self, config):
        """Default loader with base config."""
        return DocumentLoaderAWSTextract(config)

    @pytest.fixture
    def test_file_path(self):
        current_dir = os.getcwd()
        return os.path.join(current_dir, 'tests', 'files', 'invoice.pdf')

    def test_textract_config_validation(self):
        """Test TextractConfig validation for feature types."""
        # Test valid feature types
        valid_config = TextractConfig(
            feature_types=["TABLES", "FORMS"]
        )
        assert valid_config.feature_types == ["TABLES", "FORMS"]

        # Test empty feature types (Raw Text)
        raw_text_config = TextractConfig()
        assert raw_text_config.feature_types == []

        # Test invalid feature type
        with pytest.raises(ValueError) as exc_info:
            TextractConfig(feature_types=["INVALID", "TABLES"])
        assert "Invalid feature type(s)" in str(exc_info.value)
        assert "INVALID" in str(exc_info.value)

        # Test QUERIES not allowed
        with pytest.raises(ValueError) as exc_info:
            TextractConfig(feature_types=["QUERIES"])
        assert "Invalid feature type(s)" in str(exc_info.value)
        assert "QUERIES" in str(exc_info.value)

    def test_textract_raw_text(self, config, test_file_path):
        """Test raw text extraction (no feature types)."""
        loader = DocumentLoaderAWSTextract(config)  # Default is raw text
        pages = loader.load(test_file_path)
        
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert "content" in pages[0]
        assert isinstance(pages[0]["content"], str)
        assert len(pages[0]["content"]) > 0  # Should have some content

    def test_textract_with_tables(self, config, test_file_path):
        """Test extraction with TABLES feature."""
        config.feature_types = ["TABLES"]
        loader = DocumentLoaderAWSTextract(config)
        pages = loader.load(test_file_path)
        
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert "content" in pages[0]
        assert "tables" in pages[0]
        # Should have tables since TABLES feature type is enabled
        if pages[0]["tables"]:  # Only assert if the document actually contains tables
            assert isinstance(pages[0]["tables"], list)
            assert len(pages[0]["tables"]) > 0

    def test_textract_with_forms(self, config, test_file_path):
        """Test extraction with FORMS feature."""
        config.feature_types = ["FORMS"]
        loader = DocumentLoaderAWSTextract(config)
        pages = loader.load(test_file_path)
        
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert "content" in pages[0]
        assert "forms" in pages[0]
        
        # Forms should be a dictionary
        assert isinstance(pages[0]["forms"], dict)
        
        # If the test file contains forms, verify their structure
        if pages[0]["forms"]:
            # Get first form field
            first_key = next(iter(pages[0]["forms"]))
            first_value = pages[0]["forms"][first_key]
            
            # Verify form field structure
            assert isinstance(first_key, str)
            assert isinstance(first_value, str)
            assert len(first_key) > 0
            
            # Forms should not contain any None or empty values
            for key, value in pages[0]["forms"].items():
                assert key is not None and key != ""
                assert value is not None
        
        # Verify tables are empty when only FORMS feature is enabled
        assert "tables" in pages[0]
        assert pages[0]["tables"] == []

    def test_textract_forms_with_tables(self, config, test_file_path):
        """Test extraction with both FORMS and TABLES features."""
        config.feature_types = ["FORMS", "TABLES"]
        loader = DocumentLoaderAWSTextract(config)
        pages = loader.load(test_file_path)
        
        assert isinstance(pages, list)
        assert len(pages) > 0
        
        # Check forms
        assert "forms" in pages[0]
        assert isinstance(pages[0]["forms"], dict)
        
        # Check tables
        assert "tables" in pages[0]
        assert isinstance(pages[0]["tables"], list)
        
        # If we have tables, verify their structure
        for table in pages[0]["tables"]:
            assert isinstance(table, list)  # Table should be list of rows
            if table:  # If table has rows
                assert isinstance(table[0], list)  # Row should be list of cells
                
        # If we have forms, verify no empty keys
        for key, value in pages[0]["forms"].items():
            assert key is not None and key != ""
            assert value is not None

    def test_textract_raw_text_has_empty_forms(self, config, test_file_path):
        """Test that raw text extraction has empty forms dictionary."""
        # No feature types = raw text
        loader = DocumentLoaderAWSTextract(config)
        pages = loader.load(test_file_path)
        
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert "forms" in pages[0]
        assert isinstance(pages[0]["forms"], dict)
        assert len(pages[0]["forms"]) == 0  # Forms should be empty for raw text

    def test_textract_with_multiple_features(self, config, test_file_path):
        """Test extraction with multiple features."""
        config.feature_types = ["TABLES", "FORMS", "LAYOUT"]
        loader = DocumentLoaderAWSTextract(config)
        pages = loader.load(test_file_path)
        
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert "content" in pages[0]
        assert "tables" in pages[0]

    def test_vision_mode(self, loader, test_file_path):
        """Test vision mode with the new config-based loader."""
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

    def test_from_client_factory(self, config):
        """Test the from_client factory method with config."""
        # Create a mock client
        class MockClient:
            def analyze_document(self, *args, **kwargs):
                return {"Blocks": []}

        mock_client = MockClient()
        loader = DocumentLoaderAWSTextract.from_client(mock_client)
        
        assert loader.textract_client == mock_client
        assert isinstance(loader.config, TextractConfig)

    def test_legacy_initialization(self, test_file_path):
        """Test old-style (legacy) initialization and functionality."""
        # Test basic initialization
        loader = DocumentLoaderAWSTextract(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION_NAME")
        )
        assert isinstance(loader.config, TextractConfig)
        assert loader.config.feature_types == []  # Default to raw text
        
        # Test full initialization with custom parameters
        loader_full = DocumentLoaderAWSTextract(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION_NAME"),
            content="test_content",
            cache_ttl=600,
            feature_types=["TABLES", "FORMS"]
        )
        assert isinstance(loader_full.config, TextractConfig)
        assert loader_full.config.content == "test_content"
        assert loader_full.config.cache_ttl == 600
        assert loader_full.config.feature_types == ["TABLES", "FORMS"]
        
        # Test actual document processing with legacy initialization
        pages = loader_full.load(test_file_path)
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert "content" in pages[0]
        assert "tables" in pages[0]
        assert "forms" in pages[0]
        assert isinstance(pages[0]["content"], str)
        assert isinstance(pages[0]["tables"], list)
        assert isinstance(pages[0]["forms"], dict)


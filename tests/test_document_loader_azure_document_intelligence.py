import os
import pytest
from dotenv import load_dotenv
from extract_thinker.document_loader.document_loader_azure_document_intelligence import (
    DocumentLoaderAzureForm,
    AzureConfig
)
from tests.test_document_loader_base import BaseDocumentLoaderTest

load_dotenv()

class TestDocumentLoaderAzureForm(BaseDocumentLoaderTest):
    @pytest.fixture
    def config(self):
        """Base config fixture with Azure credentials."""
        return AzureConfig(
            subscription_key=os.getenv("AZURE_SUBSCRIPTION_KEY"),
            endpoint=os.getenv("AZURE_ENDPOINT")
        )

    @pytest.fixture
    def loader(self, config):
        """Default loader with base config."""
        return DocumentLoaderAzureForm(config)

    @pytest.fixture
    def test_file_path(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, 'test_images', 'invoice.png')
    
    def test_azure_config_validation(self):
        """Test AzureConfig validation for model IDs."""
        # Test default model ID (layout)
        valid_config = AzureConfig(
            subscription_key="test",
            endpoint="test"
        )
        assert valid_config.model_id == "prebuilt-layout"
        assert valid_config.is_general_model
        assert not valid_config.is_specialized_model

        # Test all general models
        for model_id in AzureConfig.GENERAL_MODELS:
            config = AzureConfig(
                subscription_key="test",
                endpoint="test",
                model_id=model_id
            )
            assert config.is_general_model
            assert not config.is_specialized_model

        # Test a specialized model
        invoice_config = AzureConfig(
            subscription_key="test",
            endpoint="test",
            model_id="prebuilt-invoice"
        )
        assert invoice_config.is_specialized_model
        assert not invoice_config.is_general_model

        # Test invalid model ID
        with pytest.raises(ValueError) as exc_info:
            AzureConfig(
                subscription_key="test",
                endpoint="test",
                model_id="invalid-model"
            )
        assert "Invalid model ID" in str(exc_info.value)
        assert "invalid-model" in str(exc_info.value)

    def test_azure_config_advanced_features(self):
        """Test AzureConfig validation for advanced features."""
        # Test valid features
        config_with_features = AzureConfig(
            subscription_key="test",
            endpoint="test",
            features=["ocrHighResolution", "formulas", "barcodes"]
        )
        assert config_with_features.has_high_resolution_ocr
        assert config_with_features.has_formula_extraction
        assert not config_with_features.has_font_extraction
        assert config_with_features.has_barcode_extraction

        # Test all valid features
        all_features = [
            "ocrHighResolution",
            "formulas", 
            "styleFont",
            "barcodes",
            "languages",
            "keyValuePairs",
            "queryFields",
            "searchablePDF"
        ]
        config_all_features = AzureConfig(
            subscription_key="test",
            endpoint="test",
            features=all_features
        )
        assert config_all_features.has_high_resolution_ocr
        assert config_all_features.has_formula_extraction
        assert config_all_features.has_font_extraction
        assert config_all_features.has_barcode_extraction

        # Test invalid features
        with pytest.raises(ValueError) as exc_info:
            AzureConfig(
                subscription_key="test",
                endpoint="test",
                features=["invalidFeature", "anotherInvalid"]
            )
        assert "Invalid features" in str(exc_info.value)
        assert "invalidFeature" in str(exc_info.value)

        # Test config without features
        config_no_features = AzureConfig(
            subscription_key="test",
            endpoint="test"
        )
        assert not config_no_features.has_high_resolution_ocr
        assert not config_no_features.has_formula_extraction
        assert not config_no_features.has_font_extraction
        assert not config_no_features.has_barcode_extraction

    def test_azure_high_resolution_ocr(self, config, test_file_path):
        """Test high resolution OCR feature."""
        # Configure with high resolution OCR
        config.features = ["ocrHighResolution"]
        config.model_id = "prebuilt-read"  # Use read model for OCR
        
        loader = DocumentLoaderAzureForm(config)
        
        # Verify configuration
        assert loader.config.has_high_resolution_ocr
        assert not loader.config.has_formula_extraction
        
        # Process document
        pages = loader.load(test_file_path)
        
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert "content" in pages[0]
        assert isinstance(pages[0]["content"], str)
        assert len(pages[0]["content"]) > 0
        print(f"High-res OCR extracted {len(pages[0]['content'])} characters")

    def test_azure_multiple_advanced_features(self, config, test_file_path):
        """Test multiple advanced features together."""
        # Configure with multiple features
        config.features = [
            "ocrHighResolution",
            "barcodes",
            "languages"
        ]
        config.model_id = "prebuilt-layout"  # Use layout for comprehensive analysis
        
        loader = DocumentLoaderAzureForm(config)
        
        # Verify configuration
        assert loader.config.has_high_resolution_ocr
        assert loader.config.has_barcode_extraction
        assert not loader.config.has_formula_extraction
        assert not loader.config.has_font_extraction
        
        # Process document
        pages = loader.load(test_file_path)
        
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert "content" in pages[0]
        
        # Check for advanced features in output
        page = pages[0]
        assert "content" in page
        assert "tables" in page
        assert "forms" in page
        
        # These fields may or may not be present depending on document content
        # but they should be accessible if the features are enabled
        if "barcodes" in page:
            assert isinstance(page["barcodes"], list)
        if "languages" in page:
            assert isinstance(page["languages"], list)

    def test_azure_loader_with_individual_feature_parameters(self):
        """Test loader initialization with individual feature parameters."""
        loader = DocumentLoaderAzureForm(
            subscription_key=os.getenv("AZURE_SUBSCRIPTION_KEY"),
            endpoint=os.getenv("AZURE_ENDPOINT"),
            model_id="prebuilt-read",
            features=["ocrHighResolution", "barcodes"]
        )
        
        assert isinstance(loader.config, AzureConfig)
        assert loader.config.has_high_resolution_ocr
        assert loader.config.has_barcode_extraction
        assert not loader.config.has_formula_extraction
        assert not loader.config.has_font_extraction

    def test_azure_from_credentials_with_features(self):
        """Test from_credentials class method with advanced features."""
        loader = DocumentLoaderAzureForm.from_credentials(
            subscription_key=os.getenv("AZURE_SUBSCRIPTION_KEY"),
            endpoint=os.getenv("AZURE_ENDPOINT"),
            features=["ocrHighResolution"],
            model_id="prebuilt-read"
        )
        
        assert isinstance(loader.config, AzureConfig)
        assert loader.config.has_high_resolution_ocr
        assert loader.config.model_id == "prebuilt-read"
        assert not loader.config.has_formula_extraction

    def test_available_features_constant(self):
        """Test that AVAILABLE_FEATURES constant contains expected features."""
        expected_features = {
            "ocrHighResolution",
            "formulas",
            "styleFont", 
            "barcodes",
            "languages",
            "keyValuePairs",
            "queryFields",
            "searchablePDF"
        }
        
        actual_features = set(AzureConfig.AVAILABLE_FEATURES)
        assert expected_features == actual_features

    def test_azure_read_model(self, config, test_file_path):
        """Test OCR/Read model extraction."""
        config.model_id = "prebuilt-read"
        loader = DocumentLoaderAzureForm(config)
        pages = loader.load(test_file_path)
        
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert "content" in pages[0]
        assert isinstance(pages[0]["content"], str)
        assert len(pages[0]["content"]) > 0
        # Read model focuses on text extraction
        assert "forms" in pages[0]
        assert isinstance(pages[0]["forms"], dict)
        assert len(pages[0]["forms"]) == 0  # Read model doesn't extract forms

    def test_azure_layout_model(self, config, test_file_path):
        """Test Layout model extraction."""
        config.model_id = "prebuilt-layout"
        loader = DocumentLoaderAzureForm(config)
        pages = loader.load(test_file_path)
        
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert "content" in pages[0]
        assert "tables" in pages[0]
        assert isinstance(pages[0]["tables"], list)
        # Layout model should handle tables and structure
        if pages[0]["tables"]:
            table = pages[0]["tables"][0]
            assert isinstance(table, list)

    def test_azure_document_model(self, config, test_file_path):
        """Test General Document model extraction."""
        config.model_id = "prebuilt-document"
        loader = DocumentLoaderAzureForm(config)
        pages = loader.load(test_file_path)
        
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert "content" in pages[0]
        assert "forms" in pages[0]
        assert "tables" in pages[0]
        # Document model should handle both key-value pairs and tables
        assert isinstance(pages[0]["forms"], dict)
        assert isinstance(pages[0]["tables"], list)

    def test_specialized_model(self, config, test_file_path):
        """Test a specialized model (invoice as example)."""
        config.model_id = "prebuilt-invoice"
        loader = DocumentLoaderAzureForm(config)
        pages = loader.load(test_file_path)
        
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert "content" in pages[0]
        assert "forms" in pages[0]
        assert isinstance(pages[0]["forms"], dict)

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

    def test_from_credentials_factory(self):
        """Test the from_credentials factory method."""
        # Test with a general model
        loader = DocumentLoaderAzureForm.from_credentials(
            subscription_key="test",
            endpoint="test",
            model_id="prebuilt-document"
        )
        
        assert isinstance(loader.config, AzureConfig)
        assert loader.config.model_id == "prebuilt-document"
        assert loader.config.is_general_model
        assert loader.config.subscription_key == "test"
        assert loader.config.endpoint == "test"

    def test_legacy_initialization(self, test_file_path):
        """Test old-style (legacy) initialization and functionality."""
        # Test basic initialization with default model
        loader = DocumentLoaderAzureForm(
            subscription_key=os.getenv("AZURE_SUBSCRIPTION_KEY"),
            endpoint=os.getenv("AZURE_ENDPOINT")
        )
        assert isinstance(loader.config, AzureConfig)
        assert loader.config.model_id == "prebuilt-layout"  # Default model
        
        # Test full initialization with custom parameters
        loader_full = DocumentLoaderAzureForm(
            subscription_key=os.getenv("AZURE_SUBSCRIPTION_KEY"),
            endpoint=os.getenv("AZURE_ENDPOINT"),
            content="test_content",
            cache_ttl=600,
            model_id="prebuilt-document"
        )
        assert isinstance(loader_full.config, AzureConfig)
        assert loader_full.config.content == "test_content"
        assert loader_full.config.cache_ttl == 600
        assert loader_full.config.model_id == "prebuilt-document"
        
        # Test actual document processing with legacy initialization
        pages = loader_full.load(test_file_path)
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert "content" in pages[0]
        assert "forms" in pages[0]
        assert "tables" in pages[0]
        assert isinstance(pages[0]["content"], str)
        assert isinstance(pages[0]["forms"], dict)
        assert isinstance(pages[0]["tables"], list)
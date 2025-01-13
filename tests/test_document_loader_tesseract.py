import os
import pytest
from dotenv import load_dotenv
from extract_thinker.document_loader.document_loader_tesseract import DocumentLoaderTesseract, TesseractConfig
from .test_document_loader_base import BaseDocumentLoaderTest

load_dotenv()

class TestDocumentLoaderTesseract(BaseDocumentLoaderTest):
    @pytest.fixture
    def tesseract_config(self):
        tesseract_path = os.getenv("TESSERACT_PATH")
        return TesseractConfig(
            tesseract_cmd=tesseract_path,
            lang="eng",
            psm=3,
            oem=3
        )

    @pytest.fixture
    def loader(self, tesseract_config):
        return DocumentLoaderTesseract(tesseract_config)

    @pytest.fixture
    def test_file_path(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "test_images", "invoice.png")

    def test_tesseract_specific_content(self, loader, test_file_path):
        """Test OCR-specific content extraction"""
        pages = loader.load(test_file_path)
        
        assert isinstance(pages, list)
        assert len(pages) > 0
        
        first_page = pages[0]
        assert "content" in first_page
        assert "Invoice" in first_page["content"]
        assert "0000001" in first_page["content"]

    def test_vision_mode(self, loader, test_file_path):
        """Override base class vision mode test for Tesseract-specific behavior"""
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

    def test_parallel_processing(self, loader, test_file_path):
        """Test parallel processing of multiple pages"""
        # Create a multi-page test case
        pages = loader.load(test_file_path)
        assert isinstance(pages, list)
        assert all("content" in page for page in pages)

    def test_tesseract_config_validation(self, tesseract_config):
        """Test TesseractConfig validation"""
        # Test invalid PSM
        with pytest.raises(ValueError, match="Invalid PSM value"):
            TesseractConfig(
                tesseract_cmd=tesseract_config.tesseract_cmd,
                psm=99
            )

        # Test invalid OEM
        with pytest.raises(ValueError, match="Invalid OEM value"):
            TesseractConfig(
                tesseract_cmd=tesseract_config.tesseract_cmd,
                oem=99
            )

        # Test negative timeout
        with pytest.raises(ValueError, match="Timeout must be non-negative"):
            TesseractConfig(
                tesseract_cmd=tesseract_config.tesseract_cmd,
                timeout=-1
            )

    def test_language_configuration(self, tesseract_config, test_file_path):
        """Test language configuration"""
        # Test single language
        config = TesseractConfig(
            tesseract_cmd=tesseract_config.tesseract_cmd,
            lang="eng"
        )
        loader = DocumentLoaderTesseract(config)
        pages = loader.load(test_file_path)
        assert len(pages) > 0

        # Test multiple languages
        config = TesseractConfig(
            tesseract_cmd=tesseract_config.tesseract_cmd,
            lang=["eng", "fra"]
        )
        loader = DocumentLoaderTesseract(config)
        pages = loader.load(test_file_path)
        assert len(pages) > 0

    def test_custom_config_params(self, tesseract_config, test_file_path):
        """Test custom configuration parameters"""
        config = TesseractConfig(
            tesseract_cmd=tesseract_config.tesseract_cmd,
            config_params={
                "tessedit_char_whitelist": "0123456789",  # Only recognize digits
                "tessdata_dir": "/usr/share/tesseract-ocr/4.00/tessdata"
            }
        )
        loader = DocumentLoaderTesseract(config)
        pages = loader.load(test_file_path)
        assert len(pages) > 0

    def test_legacy_initialization(self, tesseract_config, test_file_path):
        """Test legacy initialization method"""
        # Basic initialization
        loader = DocumentLoaderTesseract(
            tesseract_config.tesseract_cmd,
            lang="eng"
        )
        pages = loader.load(test_file_path)
        assert len(pages) > 0

        # Full initialization with all parameters
        loader = DocumentLoaderTesseract(
            tesseract_config.tesseract_cmd,
            isContainer=False,
            content=None,
            cache_ttl=300,
            lang="eng",
            psm=3,
            oem=3,
            config_params={"tessedit_char_whitelist": "0123456789"},
            timeout=30
        )
        pages = loader.load(test_file_path)
        assert len(pages) > 0

    def test_simple_initialization(self):
        """Test simple initialization and basic functionality without any special configurations"""
        tesseract_path = os.getenv("TESSERACT_PATH")
        # Simple initialization like before
        loader = DocumentLoaderTesseract(tesseract_path)
        
        # Use the test file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        test_file = os.path.join(current_dir, "test_images", "invoice.png")
        
        # Basic load and verify
        pages = loader.load(test_file)
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert "content" in pages[0]
        assert isinstance(pages[0]["content"], str)
        assert len(pages[0]["content"]) > 0  # Should have extracted some text
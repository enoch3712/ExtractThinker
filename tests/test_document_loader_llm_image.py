import os
import pytest
from io import BytesIO
from PIL import Image
import numpy as np
from extract_thinker.document_loader.document_loader_llm_image import DocumentLoaderLLMImage, LLMImageConfig
from .test_document_loader_base import BaseDocumentLoaderTest


class TestDocumentLoaderLLMImage(BaseDocumentLoaderTest):
    @pytest.fixture
    def test_image(self):
        """Create a test image for testing"""
        # Create a simple test image
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes

    @pytest.fixture
    def test_file_path(self, test_image, tmp_path):
        """Create a temporary test image file"""
        test_file = tmp_path / "test_image.jpg"
        test_file.write_bytes(test_image.getvalue())
        return str(test_file)

    @pytest.fixture
    def llm_config(self):
        """Default LLM Image configuration for testing"""
        return LLMImageConfig(
            max_image_size=1024 * 1024,  # 1MB
            image_format='jpeg',
            compression_quality=85
        )

    @pytest.fixture
    def loader(self, llm_config):
        return DocumentLoaderLLMImage(llm_config)

    def test_llm_image_specific_content(self, loader, test_file_path):
        """Test image content loading"""
        pages = loader.load(test_file_path)
        
        assert isinstance(pages, list)
        assert len(pages) > 0
        
        first_page = pages[0]
        assert "content" in first_page
        assert first_page["content"] == ""  # Should be empty since this is image-only
        assert "image" in first_page
        assert isinstance(first_page["image"], bytes)

    def test_image_processing(self, test_file_path):
        """Test image processing with different configurations"""
        # Test with format conversion
        config = LLMImageConfig(
            image_format='png',
            compression_quality=90
        )
        loader = DocumentLoaderLLMImage(config)
        pages = loader.load(test_file_path)
        assert len(pages) > 0
        
        # Verify PNG format
        image = Image.open(BytesIO(pages[0]["image"]))
        assert image.format == "PNG"

        # Test with size limit
        config = LLMImageConfig(
            max_image_size=1024,  # Very small size to force compression
            compression_quality=85
        )
        loader = DocumentLoaderLLMImage(config)
        pages = loader.load(test_file_path)
        assert len(pages) > 0
        assert len(pages[0]["image"]) <= 1024

    def test_config_validation(self):
        """Test configuration validation"""
        # Test invalid compression quality
        with pytest.raises(ValueError, match="compression_quality must be between 0 and 100"):
            LLMImageConfig(compression_quality=101)

        # Test invalid max_image_size
        with pytest.raises(ValueError, match="max_image_size must be positive"):
            LLMImageConfig(max_image_size=-1)

        # Test invalid image format
        with pytest.raises(ValueError, match="image_format must be one of"):
            LLMImageConfig(image_format="gif")

    def test_stream_loading(self, test_image):
        """Test loading from BytesIO stream"""
        loader = DocumentLoaderLLMImage()
        pages = loader.load(test_image)
        
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert "content" in pages[0]
        assert "image" in pages[0]
        assert isinstance(pages[0]["image"], bytes)

    def test_simple_initialization(self, test_file_path):
        """Test simple initialization and basic functionality without any special configurations"""
        # Simple initialization like before
        loader = DocumentLoaderLLMImage()
        
        # Basic load and verify
        pages = loader.load(test_file_path)
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert "content" in pages[0]
        assert pages[0]["content"] == ""  # Should be empty since this is image-only
        assert "image" in pages[0]
        assert isinstance(pages[0]["image"], bytes)
        
        # Test with a BytesIO stream
        with open(test_file_path, 'rb') as f:
            stream = BytesIO(f.read())
            pages = loader.load(stream)
            assert len(pages) > 0
            assert isinstance(pages[0]["image"], bytes) 

    def test_vision_mode(self, loader, test_file_path):
        """Test vision mode functionality"""
        # Verify that vision mode is enabled by default
        assert loader.vision_mode is True
        
        # Verify that can_handle_vision returns True for valid image
        assert loader.can_handle_vision(test_file_path) is True
        
        # Test with BytesIO stream
        with open(test_file_path, 'rb') as f:
            stream = BytesIO(f.read())
            assert loader.can_handle_vision(stream) is True
            
        # Test loading in vision mode
        pages = loader.load(test_file_path)
        assert len(pages) > 0
        assert "image" in pages[0]
        assert isinstance(pages[0]["image"], bytes)
        
        # Verify that the image is actually valid
        image = Image.open(BytesIO(pages[0]["image"]))
        assert image.format in ["JPEG", "PNG"]  # Should be one of our supported formats 
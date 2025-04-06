import os
import pytest
import tempfile
import io
import time
from io import BytesIO
from extract_thinker.document_loader.document_loader_mistral_ocr import DocumentLoaderMistralOCR, MistralOCRConfig
from unittest.mock import patch, Mock

class TestDocumentLoaderMistralOCR:
    def test_config_validation(self):
        """Test configuration validation"""
        # Test missing API key
        with pytest.raises(ValueError, match="api_key is required"):
            MistralOCRConfig(api_key="")

        # Test invalid cache_ttl
        with pytest.raises(ValueError, match="cache_ttl must be positive"):
            MistralOCRConfig(api_key="test_key", cache_ttl=0)

    def test_url_processing(self):
        """Test processing URLs"""
        # Skip if no API key is set
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            pytest.skip("MISTRAL_API_KEY environment variable not set")
            
        # Create config and loader
        config = MistralOCRConfig(
            api_key=api_key,
            model="mistral-ocr-latest"
        )
        loader = DocumentLoaderMistralOCR(config)
        
        # Use a publicly accessible PDF URL for testing
        test_url = "https://arxiv.org/pdf/2503.24339"
        
        # Mock the convert_to_images method to avoid Playwright issues
        with patch.object(loader, 'convert_to_images', return_value={}):
            # Mock the API response to avoid rate limiting issues
            mock_response = Mock()
            mock_response.json.return_value = {"pages": [{"markdown": "Test content", "index": 0}]}
            with patch('requests.post', return_value=mock_response) as mock_post:
                # Test URL handling
                result = loader.load(test_url)
                
                # Verify response structure
                assert isinstance(result, list)
                assert len(result) > 0
                assert "content" in result[0]
                assert isinstance(result[0]["content"], str)
                assert len(result[0]["content"]) > 0

    def test_file_processing(self):
        """Test processing local files"""
        # Skip if no API key is set
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            pytest.skip("MISTRAL_API_KEY environment variable not set")
            
        # Create config and loader
        config = MistralOCRConfig(
            api_key=api_key,
            model="mistral-ocr-latest"
        )
        loader = DocumentLoaderMistralOCR(config)
        
        # Test file path
        test_file_path = os.path.join(os.getcwd(), 'tests', 'files', 'invoice.pdf')
        
        # Verify the file exists
        if not os.path.exists(test_file_path):
            pytest.skip(f"Test file not found: {test_file_path}")
            
        # Check file size - Mistral has a 50MB limit
        file_size_mb = os.path.getsize(test_file_path) / (1024 * 1024)
        if file_size_mb > 50:
            pytest.skip(f"Test file too large ({file_size_mb:.2f}MB) - Mistral has a 50MB limit")
        
        # Mock the convert_to_images method since we're just testing API calls
        with patch.object(loader, 'convert_to_images', return_value={0: b'dummy_image_data'}):
            # Mock the API response to avoid rate limiting issues
            mock_response = Mock()
            mock_response.json.return_value = {"pages": [{"markdown": "Test content", "index": 0}]}
            
            # Mock _upload_file_to_mistral and _get_signed_url to avoid real API calls
            with patch.object(loader, '_upload_file_to_mistral', return_value='dummy-file-id'):
                with patch.object(loader, '_get_signed_url', return_value='https://example.com/file'):
                    with patch('requests.post', return_value=mock_response):
                        result = loader.load(test_file_path)
                        
                        # Verify response structure
                        assert isinstance(result, list)
                        assert len(result) > 0
                        assert "content" in result[0]
                        assert isinstance(result[0]["content"], str)
                        assert len(result[0]["content"]) > 0

    def test_image_processing(self):
        """Test processing image files"""
        # Skip if no API key is set
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            pytest.skip("MISTRAL_API_KEY environment variable not set")
            
        # Create config and loader
        config = MistralOCRConfig(
            api_key=api_key,
            model="mistral-ocr-latest"
        )
        loader = DocumentLoaderMistralOCR(config)
        
        # Create a simple test image in memory
        # This avoids issues with file path dependencies
        image_data = self._create_test_image()
        image_buffer = BytesIO(image_data)
        
        # Mock the convert_to_images method for consistent testing
        with patch.object(loader, 'convert_to_images', return_value={0: image_data}):
            # Mock the API response for image processing
            mock_response = Mock()
            mock_response.json.return_value = {"pages": [{"markdown": "Test OCR Image", "index": 0}]}
            
            # Mock _upload_file_to_mistral and _get_signed_url to avoid real API calls
            with patch.object(loader, '_upload_file_to_mistral', return_value='dummy-file-id'):
                with patch.object(loader, '_get_signed_url', return_value='https://example.com/file'):
                    with patch('requests.post', return_value=mock_response):
                        result = loader.load(image_buffer)
                        
                        # Verify response structure
                        assert isinstance(result, list)
                        assert len(result) > 0
                        assert "content" in result[0]
                        assert isinstance(result[0]["content"], str)

    def test_bytesio_processing(self):
        """Test processing BytesIO objects"""
        # Skip if no API key is set
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            pytest.skip("MISTRAL_API_KEY environment variable not set")
            
        # Create config and loader
        config = MistralOCRConfig(
            api_key=api_key,
            model="mistral-ocr-latest"
        )
        loader = DocumentLoaderMistralOCR(config)
        
        # Test file path
        test_file_path = os.path.join(os.getcwd(), 'tests', 'files', 'invoice.pdf')
        
        # Verify the file exists
        if not os.path.exists(test_file_path):
            pytest.skip(f"Test file not found: {test_file_path}")
            
        # Check file size - Mistral has a 50MB limit
        file_size_mb = os.path.getsize(test_file_path) / (1024 * 1024)
        if file_size_mb > 50:
            pytest.skip(f"Test file too large ({file_size_mb:.2f}MB) - Mistral has a 50MB limit")
        
        # Create a sample PDF BytesIO
        with open(test_file_path, 'rb') as f:
            bytes_io = BytesIO(f.read())
        
        # Mock convert_to_images for consistent testing
        with patch.object(loader, 'convert_to_images', return_value={0: b'dummy_image_data'}):
            # Mock the API response to avoid rate limiting issues
            mock_response = Mock()
            mock_response.json.return_value = {"pages": [{"markdown": "Test content", "index": 0}]}
            
            # Mock _upload_file_to_mistral and _get_signed_url to avoid real API calls
            with patch.object(loader, '_upload_file_to_mistral', return_value='dummy-file-id'):
                with patch.object(loader, '_get_signed_url', return_value='https://example.com/file'):
                    with patch('requests.post', return_value=mock_response):
                        result = loader.load(bytes_io)
                        
                        # Verify response structure
                        assert isinstance(result, list)
                        assert len(result) > 0
                        assert "content" in result[0]
                        assert isinstance(result[0]["content"], str)
                        assert len(result[0]["content"]) > 0

    def _create_test_image(self):
        """Create a simple test image for testing"""
        try:
            from PIL import Image, ImageDraw
            
            # Create a blank image with text
            img = Image.new('RGB', (200, 100), color='white')
            d = ImageDraw.Draw(img)
            d.text((20, 40), "Test OCR Image", fill='black')
            
            # Save to BytesIO
            img_byte_array = BytesIO()
            img.save(img_byte_array, format="JPEG")
            return img_byte_array.getvalue()
        except ImportError:
            # If PIL is not available, return a minimal valid JPEG
            # This is a 1x1 pixel JPEG
            return bytes([
                0xFF, 0xD8,                      # SOI marker
                0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x01, 0x00, 0x48, 0x00, 0x48, 0x00, 0x00,  # APP0 marker
                0xFF, 0xDB, 0x00, 0x43, 0x00,    # DQT marker
                0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09, 0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12, 0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20, 0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29, 0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32, 0x3C, 0x2E, 0x33, 0x34, 0x32,
                0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x11, 0x00,  # SOF marker (1x1 px)
                0xFF, 0xC4, 0x00, 0x14, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09,  # DHT marker
                0xFF, 0xC4, 0x00, 0x14, 0x10, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  # DHT marker
                0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x00, 0x00,  # SOS marker
                0xFF, 0xD9                       # EOI marker
            ])

    def test_can_handle(self):
        """Test can_handle method"""
        # Create a basic loader
        config = MistralOCRConfig(api_key="dummy_key")
        loader = DocumentLoaderMistralOCR(config)
        
        # Test URL handling
        assert loader.can_handle("https://example.com/document.pdf") is True
        
        # Test supported file formats
        for fmt in loader.SUPPORTED_FORMATS:
            # Skip actual file check by mocking _is_url and os.path.isfile
            loader._is_url = lambda x: False
            original_isfile = os.path.isfile
            os.path.isfile = lambda x: True
            
            try:
                assert loader.can_handle(f"document.{fmt}") is True
            finally:
                os.path.isfile = original_isfile
        
        # Test BytesIO handling
        assert loader.can_handle(BytesIO(b"sample content")) is True

    def test_pagination_support(self):
        """Test pagination support detection"""
        # Create a basic loader
        config = MistralOCRConfig(api_key="dummy_key")
        loader = DocumentLoaderMistralOCR(config)
        
        # Only PDF documents support pagination
        loader._is_url = lambda x: False
        original_isfile = os.path.isfile
        os.path.isfile = lambda x: True
        
        try:
            assert loader.can_handle_paginate("document.pdf") is True
            assert loader.can_handle_paginate("document.jpg") is False
            assert loader.can_handle_paginate("document.png") is False
        finally:
            os.path.isfile = original_isfile

if __name__ == "__main__":
    test = TestDocumentLoaderMistralOCR()
    test.test_url_processing()
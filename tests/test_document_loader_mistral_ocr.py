import os
import pytest
import tempfile
from io import BytesIO
from extract_thinker.document_loader.document_loader_mistral_ocr import DocumentLoaderMistralOCR, MistralOCRConfig

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
            model="mistral-ocr-latest",
            include_image_base64=True
        )
        loader = DocumentLoaderMistralOCR(config)
        
        # Use a publicly accessible PDF URL for testing
        test_url = "https://arxiv.org/pdf/2503.24339"
        
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
            model="mistral-ocr-latest",
            include_image_base64=True
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
        
        # Test file handling
        try:
            result = loader.load(test_file_path)
            
            # Verify response structure
            assert isinstance(result, list)
            assert len(result) > 0
            assert "content" in result[0]
            assert isinstance(result[0]["content"], str)
            assert len(result[0]["content"]) > 0
        except ValueError as e:
            if "Mistral API upload error" in str(e) or "Mistral API signed URL error" in str(e):
                pytest.skip(f"Mistral API upload failed: {str(e)}")
            else:
                raise

    def test_bytesio_processing(self):
        """Test processing BytesIO objects"""
        # Skip if no API key is set
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            pytest.skip("MISTRAL_API_KEY environment variable not set")
            
        # Create config and loader
        config = MistralOCRConfig(
            api_key=api_key,
            model="mistral-ocr-latest",
            include_image_base64=True
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
        
        # Test BytesIO handling
        try:
            result = loader.load(bytes_io)
            
            # Verify response structure
            assert isinstance(result, list)
            assert len(result) > 0
            assert "content" in result[0]
            assert isinstance(result[0]["content"], str)
            assert len(result[0]["content"]) > 0
        except ValueError as e:
            if "Mistral API upload error" in str(e) or "Mistral API signed URL error" in str(e):
                pytest.skip(f"Mistral API upload failed: {str(e)}")
            else:
                raise

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
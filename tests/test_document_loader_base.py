# tests/test_document_loader_base.py
import pytest
from typing import Any, Dict
from io import BytesIO

class BaseDocumentLoaderTest:
    """Base test class for all document loader implementations"""
    
    @pytest.fixture
    def loader(self):
        """Should be implemented by each test class to return their specific loader"""
        raise NotImplementedError
        
    def test_load_content_basic(self, loader, test_file_path):
        """Test basic content loading"""
        content = loader.load(test_file_path)
        assert content is not None
        
    def test_vision_mode(self, loader, test_file_path):
        """Test vision mode functionality"""
        loader.set_vision_mode(True)
        result = loader.load(test_file_path)
        
        if loader.can_handle_vision(test_file_path):
            assert isinstance(result, dict)
            assert "images" in result
            assert len(result["images"]) > 0
            for page_num, image_data in result["images"].items():
                assert isinstance(page_num, int)
                assert isinstance(image_data, bytes)
        else:
            with pytest.raises(ValueError):
                loader.load(test_file_path)
                
    def test_cache_functionality(self, loader, test_file_path):
        """Test caching behavior"""
        # First load
        result1 = loader.load(test_file_path)
        # Second load should come from cache
        result2 = loader.load(test_file_path)
        assert result1 == result2
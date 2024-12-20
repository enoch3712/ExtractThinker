# tests/test_document_loader_base.py
import pytest
from typing import Any, Dict
from io import BytesIO
import time

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
        # First load - measure time
        start_time1 = time.perf_counter()
        result1 = loader.load(test_file_path)
        duration1 = time.perf_counter() - start_time1

        # Second load should come from cache - measure time
        start_time2 = time.perf_counter()
        result2 = loader.load(test_file_path)
        duration2 = time.perf_counter() - start_time2

        # Verify results are the same
        assert result1 == result2
        
        # Verify second load was faster (cached)
        assert duration2 < duration1, f"Cached load ({duration2:.4f}s) should be faster than first load ({duration1:.4f}s)"
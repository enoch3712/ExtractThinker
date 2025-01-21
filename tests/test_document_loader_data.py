import pytest
from io import StringIO, BytesIO
from extract_thinker.document_loader.document_loader_data import DocumentLoaderData, DataLoaderConfig

class TestDocumentLoaderData:
    @pytest.fixture
    def data_config(self):
        """Default Data configuration for testing"""
        return DataLoaderConfig(
            cache_ttl=300,
            supports_vision=True
        )

    @pytest.fixture
    def loader(self, data_config):
        return DocumentLoaderData(
            cache_ttl=data_config.cache_ttl,
            supports_vision=data_config.supports_vision
        )

    @pytest.fixture
    def test_data(self):
        return [{
            "content": "Sample text",
            "image": None
        }]

    def test_preformatted_data(self, loader, test_data):
        """Test handling of pre-formatted data"""
        pages = loader.load(test_data)
        
        assert isinstance(pages, list)
        assert len(pages) == 1
        assert pages[0]["content"] == "Sample text"
        assert "image" in pages[0]
        assert pages[0]["image"] is None

    def test_vision_support(self, loader):
        """Test vision mode handling"""
        # Vision mode should be configurable
        assert loader._supports_vision is True
        assert loader.can_handle_vision("test") is True
        
        # Create loader with vision disabled
        no_vision_loader = DocumentLoaderData(supports_vision=False)
        assert no_vision_loader._supports_vision is False
        assert no_vision_loader.can_handle_vision("test") is False

    def test_string_input(self, loader):
        """Test handling of string input"""
        text = "Hello world"
        pages = loader.load(text)
        
        assert len(pages) == 1
        assert pages[0]["content"] == text
        assert pages[0]["image"] is None

    def test_stream_input(self, loader):
        """Test handling of stream input"""
        text = "Stream content"
        stream = StringIO(text)
        pages = loader.load(stream)
        
        assert len(pages) == 1
        assert pages[0]["content"] == text
        assert pages[0]["image"] is None

    def test_vision_mode_output(self, loader):
        """Test output format in vision mode"""
        loader.set_vision_mode(True)
        pages = loader.load("test")
        assert pages[0]["image"] == []
        
        loader.set_vision_mode(False)
        pages = loader.load("test")
        assert pages[0]["image"] is None

    def test_invalid_input(self, loader):
        """Test error handling for invalid inputs"""
        with pytest.raises(ValueError):
            loader.load(123)  # Invalid type

        with pytest.raises(ValueError):
            loader.load([{"wrong_key": "value"}])  # Missing content field

        with pytest.raises(ValueError):
            loader.load([123])  # Not a dict

    def test_caching(self, loader):
        """Test that caching works"""
        test_input = "cache test"
        result1 = loader.load(test_input)
        result2 = loader.load(test_input)
        
        assert result1 == result2
        assert id(result1) == id(result2)  # Should be same object (cached) 
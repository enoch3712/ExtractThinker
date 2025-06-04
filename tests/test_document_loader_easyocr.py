import os
import pytest
from io import BytesIO
import numpy as np
from extract_thinker.document_loader.document_loader_easy_ocr import DocumentLoaderEasyOCR, EasyOCRConfig
from .test_document_loader_base import BaseDocumentLoaderTest


class TestDocumentLoaderEasyOCR(BaseDocumentLoaderTest):
    @pytest.fixture
    def easyocr_config(self):
        return EasyOCRConfig(
            lang_list=['en'],
            gpu=False,
            cache_ttl=300,
            download_enabled=True,
        )

    @pytest.fixture
    def loader(self, easyocr_config):
        return DocumentLoaderEasyOCR(easyocr_config)

    @pytest.fixture
    def test_file_path(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "test_images", "invoice.png")

    def test_load_content(self, loader, test_file_path):
        """Tests that the loader can process an image file and return OCR results
            in the expected structure"""
        content = loader.load(test_file_path)
        assert isinstance(content, list) and len(content) > 0
        for page in content:
            # Each page should be a list of OCR results
            assert isinstance(page, list)
            for item in page:
                # Each OCR result should be a dictionary
                assert isinstance(item, dict)
                assert all(key in item for key in ['text', 'probability', 'bbox'])
                assert isinstance(item['text'], str)
                assert isinstance(item['probability'], (float, np.float64))
                assert isinstance(item['bbox'], (list, tuple))

    def test_load_from_bytesio(self, loader, test_file_path):
        """Tests that the loader can process an image provided as a BytesIO stream."""
        with open(test_file_path, "rb") as f:
            image_bytes = BytesIO(f.read())
        content = loader.load(image_bytes)
        assert isinstance(content, list) and len(content) > 0

    def test_can_handle(self, loader, tmp_path):
        """Tests that the loader correctly identifies supported and unsupported file formats"""
        # Supported extensions
        for ext in loader.SUPPORTED_FORMATS:
            f = tmp_path / f"file.{ext}"
            f.touch()
            assert loader.can_handle(str(f))
        # Unsupported extension
        assert not loader.can_handle(str(tmp_path / "file.abc"))
        # Missing extension
        assert not loader.can_handle(str(tmp_path / "file"))
        # BytesIO stream
        assert loader.can_handle(BytesIO(b"data"))

    def test_vision_mode(self, loader):
        """Test that vision mode is not supported"""
        # Vision mode should be disabled by default
        assert loader.vision_mode is False
        
        # Attempting to enable vision mode should raise an error
        with pytest.raises(ValueError, match="Vision mode is not supported"):
            loader.set_vision_mode(True)
        
        # Vision mode should still be False after failed attempt
        assert loader.vision_mode is False
        
        # can_handle_vision should always return False
        assert loader.can_handle_vision("test.txt") is False

    def test_language_configuration(self, test_file_path):
        """test that the loader can handle english language"""
        loader = DocumentLoaderEasyOCR(EasyOCRConfig(lang_list=['en']))
        pages = loader.load(test_file_path)
        assert len(pages) > 0
        """test that the loader can handle multiple languages(english and spanish)"""
        loader = DocumentLoaderEasyOCR(EasyOCRConfig(lang_list=['en', 'es']))
        pages = loader.load(test_file_path)
        assert len(pages) > 0

    def test_easyocr_config_validation(self):
        """Test EasyOCRConfig validation"""
        # raise error if lang_list is empty
        with pytest.raises(ValueError, match="lang_list must contain at least one"):
            EasyOCRConfig(lang_list=[])
        # raise error if cache_ttl is negative
        with pytest.raises(ValueError, match="cache_ttl must be positive"):
            EasyOCRConfig(cache_ttl=0)


  
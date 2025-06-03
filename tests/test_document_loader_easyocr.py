import os
import pytest
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
        content = loader.load(test_file_path)
        assert isinstance(content, list) and len(content) > 0
        for page in content:
            assert isinstance(page, list)
            for item in page:
                assert isinstance(item, dict)
                assert all(key in item for key in ['text', 'probability', 'bbox'])
                assert isinstance(item['text'], str)
                assert isinstance(item['probability'], (float, np.float64))
                assert isinstance(item['bbox'], (list, tuple))

    def test_can_handle_formats(self, loader, tmp_path):
        for fmt in loader.SUPPORTED_FORMATS:
            test_file = tmp_path / f"test.{fmt}"
            test_file.touch()
            assert loader.can_handle(str(test_file))

        bad_file = tmp_path / "test.xyz"
        bad_file.touch()
        assert not loader.can_handle(str(bad_file))

    def test_language_configuration(self, test_file_path):
        loader = DocumentLoaderEasyOCR(EasyOCRConfig(lang_list=['en']))
        pages = loader.load(test_file_path)
        assert len(pages) > 0

        loader = DocumentLoaderEasyOCR(EasyOCRConfig(lang_list=['en', 'es']))
        pages = loader.load(test_file_path)
        assert len(pages) > 0

    def test_simple_initialization_easyocr(self):
        config = EasyOCRConfig(lang_list=["en"])
        loader = DocumentLoaderEasyOCR(config)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        test_file = os.path.join(current_dir, "test_images", "invoice.png")
        pages = loader.load(test_file)
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert isinstance(pages[0], list)
        assert isinstance(pages[0][0], dict)
        assert "text" in pages[0][0]
        assert isinstance(pages[0][0]["text"], str)

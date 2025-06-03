import os
import time
import pytest
from io import BytesIO
from PIL import Image
from extract_thinker.document_loader.document_loader_easy_ocr import DocumentLoaderEasyOCR, EasyOCRConfig
from .test_document_loader_base import BaseDocumentLoaderTest
import numpy as np


class TestDocumentLoaderEasyOCR(BaseDocumentLoaderTest):
    @pytest.fixture
    def easyocr_config(self):
        return EasyOCRConfig( lang_list=['en'],
                              gpu=False, 
                              cache_ttl=300,
                              download_enabled=True)

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
            # Each page should be a list of OCR results
            assert isinstance(page, list)
            for item in page:
                # Each OCR result should be a dict with required fields
                assert isinstance(item, dict)
                assert all(key in item for key in ['text', 'probability', 'bbox'])
                assert isinstance(item['text'], str)
                assert isinstance(item['probability'], (float, np.float64))
                assert isinstance(item['bbox'], (list, tuple))

    def test_load_content_bytesio(self, loader, test_file_path):
        with open(test_file_path, 'rb') as f:
            stream = BytesIO(f.read())
        content = loader.load(stream)
        assert isinstance(content, list) and len(content) > 0

    def test_vision_mode_enabled(self, loader, test_file_path):
        loader.set_vision_mode(True)
        result = loader.load(test_file_path)
        assert isinstance(result, dict)
         # Check that 'images' key exists and is a dictionary
        assert "images" in result
        images = result["images"]
        assert isinstance(images, dict)
        for page_num, image_data in images.items():
            assert isinstance(page_num, int)
            assert isinstance(image_data, bytes)
        assert "pages" in result
        assert isinstance(result["pages"], list)

    def test_can_handle_formats(self, loader, tmp_path):
        for fmt in loader.SUPPORTED_FORMATS:
            test_file = tmp_path / f"test.{fmt}"
            test_file.touch()
            assert loader.can_handle(str(test_file))

        bad_file = tmp_path / "test.xyz"
        bad_file.touch()
        assert not loader.can_handle(str(bad_file))

        assert loader.can_handle(BytesIO(b"dummy data"))

    def test_caching_behavior(self, loader, test_file_path):
        t1 = time.perf_counter()
        result1 = loader.load(test_file_path)
        t2 = time.perf_counter()
        result2 = loader.load(test_file_path)
        t3 = time.perf_counter()

        assert result1 == result2
        # Second load should be faster due to caching
        assert (t3 - t2) < (t2 - t1)

    def test_pdf_handling(self, test_file_path):
        pdf_path = test_file_path.replace('.png', '.pdf')
        Image.open(test_file_path).save(pdf_path, 'PDF')

        try:
            loader = DocumentLoaderEasyOCR(EasyOCRConfig(lang_list=['en'], gpu=False))
            content = loader.load(pdf_path)
            assert isinstance(content, list) and len(content) > 0
        finally:
            os.remove(pdf_path)

    def test_language_configuration(self, test_file_path):
    # Test one language
      loader = DocumentLoaderEasyOCR(EasyOCRConfig(lang_list=['en']))
      pages = loader.load(test_file_path)
      assert len(pages) > 0

    # Test multiple languages
      loader = DocumentLoaderEasyOCR(EasyOCRConfig(lang_list=['en', 'es']))
      pages = loader.load(test_file_path)
      assert len(pages) > 0


    def test_simple_initialization_easyocr(self):
      #Test simple initialization and basic functionality for EasyOCR"
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

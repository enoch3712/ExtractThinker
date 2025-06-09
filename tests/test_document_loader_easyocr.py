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
            # Each page should be a dictionary with 'content'
            assert isinstance(page, dict)
            assert "content" in page

    def test_load_from_bytesio(self, loader, test_file_path):
        """Tests that the loader can process an image from a BytesIO stream"""
        with open(test_file_path, "rb") as f:
            image_bytes = BytesIO(f.read())
        content = loader.load(image_bytes)
        assert isinstance(content, list) and len(content) > 0
        for page in content:
             # Each page should be a dictionary with 'content'
            assert isinstance(page, dict)
            assert "content" in page

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

    def test_unsupported_file_extension(self, loader):
        """Tests that an unsupported file extension returns False from can_handle"""
        # ... existing code ...


@pytest.fixture
def easy_ocr_loader_bbox():
    config = EasyOCRConfig(include_bbox=True)
    return DocumentLoaderEasyOCR(config)

@pytest.fixture
def easy_ocr_loader_no_bbox():
    config = EasyOCRConfig(include_bbox=False)
    return DocumentLoaderEasyOCR(config)

def test_load_pdf_with_bbox(easy_ocr_loader_bbox):
    """Test loading a PDF file with bounding box information."""
    file_path = os.path.join(os.getcwd(), "tests", "files", "invoice.pdf")
    result = easy_ocr_loader_bbox.load(file_path)
    
    assert isinstance(result, list)
    assert len(result) > 0  # Assuming the PDF has at least one page
    
    first_page = result[0]
    assert "content" in first_page
    assert "detail" in first_page
    assert isinstance(first_page["content"], str)
    assert isinstance(first_page["detail"], list)
    
    if len(first_page["detail"]) > 0:
        first_detail = first_page["detail"][0]
        assert "bbox" in first_detail
        assert "text" in first_detail
        assert "probability" in first_detail

def test_load_pdf_without_bbox(easy_ocr_loader_no_bbox):
    """Test loading a PDF file without bounding box information."""
    file_path = os.path.join(os.getcwd(), "tests", "files", "invoice.pdf")
    result = easy_ocr_loader_no_bbox.load(file_path)
    
    assert isinstance(result, list)
    assert len(result) > 0
    
    first_page = result[0]
    assert "content" in first_page
    assert "detail" not in first_page
    assert isinstance(first_page["content"], str)

def test_load_image_with_bbox(easy_ocr_loader_bbox):
    """Test loading an image file with bounding box information."""
    # Using a known image file from the tests directory
    file_path = os.path.join(os.getcwd(), "tests", "test_images", "invoice.png")
    result = easy_ocr_loader_bbox.load(file_path)
    
    assert isinstance(result, list)
    assert len(result) == 1
    
    page = result[0]
    assert "content" in page
    assert "detail" in page
    assert isinstance(page["content"], str)
    assert isinstance(page["detail"], list)

def test_load_image_without_bbox(easy_ocr_loader_no_bbox):
    """Test loading an image file without bounding box information."""
    file_path = os.path.join(os.getcwd(), "tests", "test_images", "invoice.png")
    result = easy_ocr_loader_no_bbox.load(file_path)
    
    assert isinstance(result, list)
    assert len(result) == 1
    
    page = result[0]
    assert "content" in page
    assert "detail" not in page
    assert isinstance(page["content"], str)

def test_unsupported_file_type(easy_ocr_loader_no_bbox):
    """Test that an unsupported file type raises a ValueError."""
    file_path = os.path.join(os.getcwd(), "README.md")
    with pytest.raises(ValueError):
        easy_ocr_loader_no_bbox.load(file_path)

def test_load_multipage_pdf(easy_ocr_loader_no_bbox):
    """Test loading a multi-page PDF and verify the page count."""
    file_path = os.path.join(os.getcwd(), "tests", "files", "bulk.pdf")
    result = easy_ocr_loader_no_bbox.load(file_path)
    
    assert isinstance(result, list)
    assert len(result) == 3
    
    for page in result:
        assert "content" in page
        assert isinstance(page["content"], str)



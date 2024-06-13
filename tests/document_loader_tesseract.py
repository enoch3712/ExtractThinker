from io import BytesIO
import os
from dotenv import load_dotenv

from extract_thinker.document_loader.document_loader_tesseract import DocumentLoaderTesseract

cwd = os.getcwd()
load_dotenv()

# Arrange
tesseract_path = os.getenv("TESSERACT_PATH")
loader = DocumentLoaderTesseract(tesseract_path)
test_file_path = os.path.join(cwd, "test_images", "invoice.png")


def test_load_content_from_file():
    # Act
    content = loader.load_content_from_file(test_file_path)

    # Assert
    assert content is not None
    assert "Invoice" in content
    assert "0000001" in content


def test_load_content_from_stream():
    with open(test_file_path, 'rb') as f:
        test_image_stream = BytesIO(f.read())

    # Act
    content = loader.load_content_from_stream(test_image_stream)

    # Assert
    assert content is not None
    assert "Invoice" in content
    assert "0000001" in content


def test_cache_for_file():
    # Act
    content1 = loader.load_content_from_file(test_file_path)
    content2 = loader.load_content_from_file(test_file_path)

    # Assert
    assert content1 is content2


def test_queue_load():
    for _ in range(10):
        # Act
        content = loader.load_content_from_file(test_file_path)
        # Assert
        assert "0000001" in content

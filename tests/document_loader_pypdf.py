from io import BytesIO
import os
from dotenv import load_dotenv

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from extract_thinker.document_loader.document_loader_pypdf import DocumentLoaderPyPdf

cwd = os.getcwd()
load_dotenv()

# Arrange
loader = DocumentLoaderPyPdf()
test_file_path = os.path.join(cwd, "tests", "files", "ip268_en_0.pdf")


def test_load_content_from_file():
    # Act
    content = loader.load_content_from_file("C:\\Users\\Lopez\\Downloads\\DMV_Combined_REG_files.pdf")

    # Assert
    assert content is not None
    assert "Invoice" in content["text"]
    assert "0000001" in content["text"]


def test_load_content_from_stream():
    with open(test_file_path, 'rb') as f:
        test_pdf_stream = BytesIO(f.read())

    # Act
    content = loader.load_content_from_stream(test_pdf_stream)

    # Assert
    assert content is not None
    assert "Invoice" in content["text"]
    assert "0000001" in content["text"]


def test_cache_for_file():
    # Act
    content1 = loader.load_content_from_file(test_file_path)
    content2 = loader.load_content_from_file(test_file_path)

    # Assert
    assert content1 is content2


test_load_content_from_file()
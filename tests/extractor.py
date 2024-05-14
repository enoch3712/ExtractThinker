
import os
from dotenv import load_dotenv

from extract_thinker.extractor import Extractor
from extract_thinker.document_loader.document_loader_tesseract import DocumentLoaderTesseract
from tests.models.invoice import InvoiceContract

load_dotenv()
cwd = os.getcwd()


def test_extract_with_tessaract_and_claude():

    # Arrange
    tesseract_path = os.getenv("TESSERACT_PATH")
    test_file_path = os.path.join(cwd, "test_images", "invoice.png")

    extractor = Extractor()
    extractor.load_document_loader(
        DocumentLoaderTesseract(tesseract_path)
    )
    extractor.load_llm("claude-3-haiku-20240307")

    # Act
    result = extractor.extract(test_file_path, InvoiceContract)

    # Assert
    assert result is not None
    assert result.invoice_number == "0000001"
    assert result.invoice_date == "2014-05-07"

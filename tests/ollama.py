import os
from dotenv import load_dotenv
from extract_thinker import LLM
from extract_thinker import DocumentLoaderTesseract

from extract_thinker.extractor import Extractor
from tests.models.invoice import InvoiceContract

load_dotenv()
cwd = os.getcwd()


def test_extract_with_ollama():

    # Arrange
    tesseract_path = os.getenv("TESSERACT_PATH")
    test_file_path = os.path.join(cwd, "test_images", "invoice.png")

    extractor = Extractor()
    extractor.load_document_loader(
        DocumentLoaderTesseract(tesseract_path)
    )

    llm = LLM("ollama/phi3", "http://localhost:11434")
    extractor.load_llm(llm)

    # Act
    result = extractor.extract(test_file_path, InvoiceContract)

    # Assert
    assert result is not None
    assert result.invoice_number == "0000001"
    assert result.invoice_date == "2014-05-07"

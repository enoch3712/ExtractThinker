import os
from dotenv import load_dotenv
from extract_thinker.extractor import Extractor
from extract_thinker.document_loader.document_loader_tesseract import DocumentLoaderTesseract
from extract_thinker.models import Classification, ClassificationResponse

load_dotenv()
cwd = os.getcwd()


def test_classify_feature():
    # Arrange
    tesseract_path = os.getenv("TESSERACT_PATH")
    test_file_path = os.path.join(cwd, "test_images", "invoice.png")

    classifications = [
        Classification(name="Driver License", description="This is a driver license"),
        Classification(name="Invoice", description="This is an invoice"),
    ]

    extractor = Extractor()
    extractor.load_document_loader(DocumentLoaderTesseract(tesseract_path))
    extractor.load_llm("claude-3-haiku-20240307")

    # Act
    result = extractor.classify_from_path(test_file_path, classifications)

    # Assert
    assert result is not None
    assert isinstance(result, ClassificationResponse)
    assert result.name == "Invoice"

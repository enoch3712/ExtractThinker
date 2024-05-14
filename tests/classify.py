import asyncio
import os
from dotenv import load_dotenv
from extract_thinker.extractor import Extractor
from extract_thinker.process import Process
from extract_thinker.document_loader.document_loader_tesseract import DocumentLoaderTesseract
from extract_thinker.models import classification, classification_response

load_dotenv()
cwd = os.getcwd()


def test_classify_feature():
    # Arrange
    tesseract_path = os.getenv("TESSERACT_PATH")
    test_file_path = os.path.join(cwd, "test_images", "invoice.png")

    classifications = [
        classification(name="Driver License", description="This is a driver license"),
        classification(name="Invoice", description="This is an invoice"),
    ]

    extractor = Extractor()
    extractor.load_document_loader(DocumentLoaderTesseract(tesseract_path))
    extractor.load_llm("claude-3-haiku-20240307")

    # Act
    result = extractor.classify_from_path(test_file_path, classifications)

    # Assert
    assert result is not None
    assert isinstance(result, classification_response)
    assert result.name == "Invoice"


def test_classify():
    # Arrange
    test_file_path = os.path.join(cwd, "test_images", "invoice.png")

    process = Process()
    tesseract_path = os.getenv("TESSERACT_PATH")

    document_loader = DocumentLoaderTesseract(tesseract_path)

    open35extractor = Extractor(document_loader)
    open35extractor.load_llm("gpt-3.5-turbo")

    mistral2extractor = Extractor(document_loader)
    mistral2extractor.load_llm("claude-3-haiku-20240307")

    gpt4extractor = Extractor(document_loader)
    gpt4extractor.load_llm("gpt-4-turbo")

    process.add_classifyExtractor([[open35extractor, mistral2extractor], [gpt4extractor]])

    classifications = [
        classification(name="Driver License", description="This is a driver license"),
        classification(name="Invoice", description="This is an invoice"),
    ]

    # Act
    result = asyncio.run(process.classify_async(test_file_path, classifications))

    # Assert
    assert result is not None
    assert isinstance(result, classification_response)
    assert result.name == "Invoice"

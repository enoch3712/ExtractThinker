import os
import asyncio
from dotenv import load_dotenv

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from extract_thinker.extractor import Extractor
from extract_thinker.process import Process, ClassificationStrategy
from extract_thinker.document_loader.document_loader_tesseract import DocumentLoaderTesseract
from extract_thinker.models.classification import Classification
from extract_thinker.models.classification_response import ClassificationResponse
from tests.models.invoice import InvoiceContract
from tests.models.driver_license import DriverLicense

# Setup environment and common paths
load_dotenv()
tesseract_path = os.getenv("TESSERACT_PATH")
CURRENT_WORKING_DIRECTORY = os.getcwd()
INVOICE_FILE_PATH = os.path.join(CURRENT_WORKING_DIRECTORY, "tests", "test_images", "invoice.png")
DRIVER_LICENSE_FILE_PATH = os.path.join(CURRENT_WORKING_DIRECTORY, "tests", "test_images", "driver_license.jpg")

# Common classifications setup
COMMON_CLASSIFICATIONS = [
    Classification(name="Driver License", description="This is a driver license"),
    Classification(name="Invoice", description="This is an invoice"),
]


def setup_extractors():
    """Sets up and returns a list of configured extractors."""
    tesseract_path = os.getenv("TESSERACT_PATH")
    document_loader = DocumentLoaderTesseract(tesseract_path)

    extractors = [
        ("gpt-3.5-turbo", "gpt-3.5-turbo"),
        ("claude-3-haiku-20240307", "claude-3-haiku-20240307"),
        ("gpt-4o", "gpt-4o")
    ]

    configured_extractors = []
    for llm_name, llm_version in extractors:
        extractor = Extractor(document_loader)
        extractor.load_llm(llm_version)
        configured_extractors.append(extractor)

    return configured_extractors


def arrange_process_with_extractors():
    """Arrange a process with predefined extractors."""
    process = Process()
    extractors = setup_extractors()
    process.add_classify_extractor([extractors[:2], [extractors[2]]])
    return process


def test_classify_feature():
    """Test classification using a single feature."""
    extractor = setup_extractors()[1]  # Using the second configured extractor
    result = extractor.classify(INVOICE_FILE_PATH, COMMON_CLASSIFICATIONS)

    assert result is not None
    assert isinstance(result, ClassificationResponse)
    assert result.name == "Invoice"


def test_classify_async():
    """Test asynchronous classification."""
    process = arrange_process_with_extractors()
    result = asyncio.run(process.classify_async(INVOICE_FILE_PATH, COMMON_CLASSIFICATIONS))

    assert result is not None
    assert isinstance(result, ClassificationResponse)
    assert result.name == "Invoice"


def test_classify_consensus():
    """Test classification using consensus strategy."""
    process = arrange_process_with_extractors()
    result = process.classify(INVOICE_FILE_PATH, COMMON_CLASSIFICATIONS, strategy=ClassificationStrategy.CONSENSUS)

    assert result is not None
    assert isinstance(result, ClassificationResponse)
    assert result.name == "Invoice"


def test_classify_higher_order():
    """Test classification using higher order strategy."""
    process = arrange_process_with_extractors()

    # Act
    result = process.classify(INVOICE_FILE_PATH, COMMON_CLASSIFICATIONS, strategy=ClassificationStrategy.HIGHER_ORDER)

    # Assert
    assert result is not None
    assert isinstance(result, ClassificationResponse)
    assert result.name == "Invoice"


def test_classify_both():
    """Test classification using both consensus and higher order strategies with a threshold."""
    process = arrange_process_with_extractors()
    result = process.classify(INVOICE_FILE_PATH, COMMON_CLASSIFICATIONS, strategy=ClassificationStrategy.BOTH, threshold=9)

    assert result is not None
    assert isinstance(result, ClassificationResponse)
    assert result.name == "Invoice"


def test_with_contract():
    """Test classification using both consensus and higher order strategies with a threshold."""
    process = arrange_process_with_extractors()

    COMMON_CLASSIFICATIONS[0].contract = InvoiceContract
    COMMON_CLASSIFICATIONS[1].contract = DriverLicense

    result = process.classify(INVOICE_FILE_PATH, COMMON_CLASSIFICATIONS, strategy=ClassificationStrategy.CONSENSUS)

    assert result is not None
    assert isinstance(result, ClassificationResponse)
    assert result.name == "Invoice"


def setup_process_with_gpt4_extractor():
    """Sets up and returns a process configured with only the GPT-4 extractor."""
    tesseract_path = os.getenv("TESSERACT_PATH")
    document_loader = DocumentLoaderTesseract(tesseract_path)

    # Initialize the GPT-4 extractor
    gpt_4_extractor = Extractor(document_loader)
    gpt_4_extractor.load_llm("gpt-4o")

    # Create the process with only the GPT-4 extractor
    process = Process([gpt_4_extractor])
    return process


def test_with_image():
    """Test classification using both consensus and higher order strategies with a threshold."""
    process = arrange_process_with_extractors()

    COMMON_CLASSIFICATIONS[0].contract = InvoiceContract
    COMMON_CLASSIFICATIONS[1].contract = DriverLicense

    COMMON_CLASSIFICATIONS[0].image = INVOICE_FILE_PATH
    COMMON_CLASSIFICATIONS[1].image = DRIVER_LICENSE_FILE_PATH

    result = process.classify(INVOICE_FILE_PATH, COMMON_CLASSIFICATIONS, strategy=ClassificationStrategy.CONSENSUS, image=True)

    assert result is not None
    assert isinstance(result, ClassificationResponse)
    assert result.name == "Invoice"


test_with_image()

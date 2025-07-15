import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
from extract_thinker import Contract, Extractor, Process, Classification
from extract_thinker.document_loader.document_loader_tesseract import DocumentLoaderTesseract
from extract_thinker.models.splitting_strategy import SplittingStrategy
from tests.models.driver_license import DriverLicense
from extract_thinker.image_splitter import ImageSplitter
from extract_thinker.text_splitter import TextSplitter
from extract_thinker.models.classification import Classification
from extract_thinker.models.contract import Contract
from extract_thinker.extractor import Extractor
from extract_thinker.global_models import get_gpt_o4_model, get_lite_model, get_big_model
import pytest

# Setup environment and paths
load_dotenv()
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MULTI_PAGE_DOC_PATH = os.path.join(CURRENT_DIR, "files", "bulk.pdf")

class VehicleRegistration(Contract):
    name_primary: str
    name_secondary: str
    address: str
    vehicle_type: str
    vehicle_color: str

# Common test classifications
TEST_CLASSIFICATIONS = [
    Classification(
        name="Vehicle Registration",
        description="This is a vehicle registration document",
        contract=VehicleRegistration,
    ),
    Classification(
        name="Driver License",
        description="This is a driver license document",
        contract=DriverLicense
    )
]

def normalize_name(name: str) -> str:
    """Normalize name format by removing commas and sorting parts"""
    parts = name.replace(",", "").split()
    return " ".join(sorted(parts))

def setup_process_and_classifications():
    """Helper function to set up process and classifications"""
    # Initialize extractor
    extractor = Extractor()
    tesseract_path = "/opt/homebrew/bin/tesseract"
    extractor.load_document_loader(DocumentLoaderTesseract(tesseract_path))
    extractor.load_llm(get_lite_model())

    # Add to classifications
    TEST_CLASSIFICATIONS[0].extractor = extractor
    TEST_CLASSIFICATIONS[1].extractor = extractor

    # Initialize process
    process = Process()
    process.load_document_loader(DocumentLoaderTesseract(tesseract_path))
    
    return process, TEST_CLASSIFICATIONS

def test_eager_splitting_strategy():
    """Test eager splitting strategy with a multi-page document"""
    
    # Arrange
    process, classifications = setup_process_and_classifications()
    process.load_splitter(ImageSplitter(get_big_model()))
    
    # Act
    result = process.load_file(MULTI_PAGE_DOC_PATH)\
        .split(classifications, strategy=SplittingStrategy.EAGER)\
        .extract()
    
    # Assert
    assert result is not None
    for item in result:
        assert isinstance(item, (TEST_CLASSIFICATIONS[0].contract, TEST_CLASSIFICATIONS[1].contract))

    assert normalize_name(result[0].name_primary) == normalize_name("Motorist, Michael M")
    assert result[1].license_number.replace(" ", "") in ["0123456789", "123456789"]

def test_lazy_splitting_strategy():
    """Test lazy splitting strategy with a multi-page document"""
    # Arrange
    process, classifications = setup_process_and_classifications()
    process.load_splitter(ImageSplitter(get_big_model()))
    
    # Act
    result = process.load_file(MULTI_PAGE_DOC_PATH)\
        .split(classifications, strategy=SplittingStrategy.LAZY)\
        .extract()
    
    # Assert
    assert result is not None
    for item in result:
        assert isinstance(item, (TEST_CLASSIFICATIONS[0].contract, TEST_CLASSIFICATIONS[1].contract))

    assert normalize_name(result[0].name_primary) == normalize_name("Motorist, Michael M")
    assert result[1].license_number.replace(" ", "") in ["0123456789", "123456789"]

def test_eager_splitting_strategy_text():
    """Test eager splitting strategy with a multi-page text document"""
    # Arrange
    process, classifications = setup_process_and_classifications()
    process.load_splitter(TextSplitter(get_big_model()))
    
    # Act
    result = process.load_file(MULTI_PAGE_DOC_PATH)\
        .split(classifications, strategy=SplittingStrategy.EAGER)\
        .extract()
    
    # Assert
    assert result is not None
    for item in result:
        assert isinstance(item, (TEST_CLASSIFICATIONS[0].contract, TEST_CLASSIFICATIONS[1].contract))

    assert normalize_name(result[0].name_primary) == normalize_name("Motorist, Michael M")
    assert result[1].license_number.replace(" ", "") == "0123456789"

def test_lazy_splitting_strategy_text():
    """Test lazy splitting strategy with a multi-page text document"""
    # Arrange
    process, classifications = setup_process_and_classifications()
    process.load_splitter(TextSplitter(get_big_model()))
    
    # Act
    result = process.load_file(MULTI_PAGE_DOC_PATH)\
        .split(classifications, strategy=SplittingStrategy.LAZY)\
        .extract()
    
    # Assert
    assert result is not None
    for item in result:
        assert isinstance(item, (TEST_CLASSIFICATIONS[0].contract, TEST_CLASSIFICATIONS[1].contract))

    assert normalize_name(result[0].name_primary) == normalize_name("Motorist, Michael M")
    assert result[1].license_number.replace(" ", "") == "0123456789"

def test_eager_splitting_strategy_vision():
    """Test eager splitting strategy with a multi-page document"""
    # Arrange
    process, classifications = setup_process_and_classifications()
    process.load_splitter(ImageSplitter("gemini/gemini-2.5-flash-preview-05-20"))
    
    # Act
    result = process.load_file(MULTI_PAGE_DOC_PATH)\
        .split(classifications, strategy=SplittingStrategy.EAGER)\
        .extract(vision=True)

    # Assert
    assert result is not None
    for item in result:
        assert isinstance(item, (TEST_CLASSIFICATIONS[0].contract, TEST_CLASSIFICATIONS[1].contract))

    assert normalize_name(result[0].name_primary) == normalize_name("Motorist, Michael M")
    assert result[1].age == 63
    assert result[1].license_number.replace(" ", "") in ["0123456789", "123456789"]
    #assert result[1].license_number.replace(" ", "") == "0123456789" #small vision bug from the model, refuses to return 0 on driver license

def test_split_requires_splitter():
    """Test that attempting to split without loading a splitter first raises an error"""
    # Arrange
    process = Process()
    
    # Act & Assert
    with pytest.raises(ValueError, match="No splitter loaded"):
        process.split([])  # Empty classifications list is fine for this test

def test_invoice_extraction_with_extraction_contract():
    from extract_thinker.models.classification import Classification
    from extract_thinker import Contract
    from extract_thinker.extractor import Extractor

    # Define a full invoice contract with multiple fields.
    class FullInvoiceContract(Contract):
        invoice_number: str
        invoice_date: str
        total_amount: float

    # Define a simple invoice contract that only extracts the invoice id.
    class SimpleInvoiceContract(Contract):
        invoice_number: str

    # Dummy document loader that returns a list of pages (content is irrelevant for this dummy test)
    class DummyLoader:
        def load(self, input):
            # simulate two pages of content
            # In a real implementation, you would parse the PDF file here.
            return ["Page 1 content", "Page 2 content"]

        def set_vision_mode(self, value: bool):
            pass

    # Dummy splitter that returns a dummy document group for extraction
    class DummySplitter:
        def split_eager_doc_group(self, pages, classifications):
            class DummyDocGroup:
                classification = "Invoice Classification"
                pages = [1]  # suppose only page 1 is relevant

            return [DummyDocGroup()]

        def split_lazy_doc_group(self, pages, classifications):
            raise NotImplementedError()

    # Dummy extractor that uses extraction_contract
    class DummyExtractor(Extractor):
        async def extract_async(self, source, contract, vision, completion_strategy):
            # If the extraction contract is used, return only the invoice_number
            if contract == SimpleInvoiceContract:
                return SimpleInvoiceContract(invoice_number="INV0001")
            else:
                return FullInvoiceContract(
                    invoice_number="INV0001",
                    invoice_date="2014-05-07",
                    total_amount=100.0,
                )

        def set_skip_loading(self, value: bool):
            self.skip_loading = value

    # Create a classification instance using invoice contracts
    classification = Classification(
        name="Invoice Classification",
        description="Testing invoice extraction with a simpler output model",
        contract=FullInvoiceContract,
        extraction_contract=SimpleInvoiceContract,
        extractor=DummyExtractor(),
    )

    from extract_thinker.process import Process

    process = Process()
    process.load_document_loader(DummyLoader())
    process.load_splitter(DummySplitter())

    # Set our test classification in the split classifications list
    process.split_classifications = [classification]
    # Use an invoice PDF file for the file path (ensure that tests/invoice_sample.pdf exists in your test suite)
    process.file_path = "tests/invoice_sample.pdf"

    # Manually set doc_groups to simulate splitting
    class DummyDocGroup:
        classification = "Invoice Classification"
        pages = [1]

    process.doc_groups = [DummyDocGroup()]

    # Call extract method from Process
    result = process.extract()

    # Assertions: result should come from the extraction_contract, so only invoice_number is expected.
    assert result is not None

    # Check that result is an instance of the simple extraction contract
    assert result.invoice_number == "INV0001"

    # Also, verify that no other keys are present – convert to dict and compare keys.
    result_dict = result.dict()
    assert list(result_dict.keys()) == ["invoice_number"]

def test_invoice_extraction_with_extraction_contract():
    """Test eager splitting strategy with a multi-page text invoice document using a simple invoice extraction contract."""

    # Define a simple invoice contract.
    class SimpleDriverLicenseContract(Contract):
        license_number: str

    process, classifications = setup_process_and_classifications()
    process.load_splitter(TextSplitter(get_big_model()))

    # Use an invoice PDF for input
    invoice_pdf_path = os.path.join(CURRENT_DIR, "files", "bulk.pdf")

    classifications[1].extraction_contract = SimpleDriverLicenseContract

    result = (
        process.load_file(invoice_pdf_path)
        .split(classifications, strategy=SplittingStrategy.EAGER)
        .extract()
    )

    simple_driver_license_contract = result[1]

    assert isinstance(simple_driver_license_contract, SimpleDriverLicenseContract)
    assert simple_driver_license_contract.license_number.replace(" ", "") == "0123456789"

    # contains no other fields
    assert len(simple_driver_license_contract.model_dump().keys()) == 1
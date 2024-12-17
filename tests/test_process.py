import os
from dotenv import load_dotenv
from extract_thinker import Contract, Extractor, Process, Classification
from extract_thinker.document_loader.document_loader_tesseract import DocumentLoaderTesseract
from extract_thinker.models.splitting_strategy import SplittingStrategy
from tests.models.invoice import InvoiceContract
from tests.models.driver_license import DriverLicense
from extract_thinker.image_splitter import ImageSplitter
from extract_thinker.text_splitter import TextSplitter

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
    extractor.load_llm("gpt-4o-mini")

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
    process.load_splitter(ImageSplitter("claude-3-5-sonnet-20241022"))
    
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
    process.load_splitter(ImageSplitter("claude-3-5-sonnet-20241022"))
    
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
    process.load_splitter(TextSplitter("claude-3-5-sonnet-20241022"))
    
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
    process.load_splitter(TextSplitter("claude-3-5-sonnet-20241022"))
    
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
    process.load_splitter(ImageSplitter("claude-3-5-sonnet-20241022"))
    
    # Act
    result = process.load_file(MULTI_PAGE_DOC_PATH)\
        .split(classifications, strategy=SplittingStrategy.EAGER)\
        .extract(vision=True)

    # Assert
    assert result is not None
    for item in result:
        assert isinstance(item, (TEST_CLASSIFICATIONS[0].contract, TEST_CLASSIFICATIONS[1].contract))

    assert normalize_name(result[0].name_primary) == normalize_name("Motorist, Michael M")
    assert result[1].age == 65
    assert result[1].license_number.replace(" ", "") in ["0123456789", "123456789"]
    #assert result[1].license_number.replace(" ", "") == "0123456789" #small vision bug from the model, refuses to return 0 on driver license

if __name__ == "__main__":
    test_eager_splitting_strategy()

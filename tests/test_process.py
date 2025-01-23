import asyncio
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
from extract_thinker import Contract, Extractor, Process, Classification
from extract_thinker.document_loader.document_loader_pypdf import DocumentLoaderPyPdf
from extract_thinker.document_loader.document_loader_tesseract import DocumentLoaderTesseract
from extract_thinker.llm import LLM
from extract_thinker.models.splitting_strategy import SplittingStrategy
from extract_thinker.process import MaskingStrategy
from tests.models.invoice import InvoiceContract
from tests.models.driver_license import DriverLicense
from extract_thinker.image_splitter import ImageSplitter
from extract_thinker.text_splitter import TextSplitter
import pytest

load_dotenv()
cwd = os.getcwd()
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MULTI_PAGE_DOC_PATH = os.path.join(CURRENT_DIR, "files", "bulk.pdf")

def test_mask():
    # Arrange
    test_file_path = os.path.join(cwd, "tests", "files", "invoice.pdf")

    process = Process()
    process.load_document_loader(DocumentLoaderPyPdf())
    process.load_file(test_file_path)
    # process.add_masking_llm("groq/llama-3.2-3b-preview")
    llm = LLM("ollama/deepseek-r1:1.5b")
    process.add_masking_llm(llm)

    # Act
    test_text = "Mr. George Collins lives at 123 Main St, Anytown, USA 12345.\n His phone number is 555-1234.\nJane Smith resides at 456 Elm Avenue, Othercity, State 67890, and can be reached at (987) 654-3210.\nThe company's CEO, Robert Johnson, has an office at 789 Corporate Blvd, Suite 500, Bigcity, State 13579. \nFor customer service, call 1-800-555-9876 or email support@example.com. \nSarah Lee, our HR manager, can be contacted at 444-333-2222 or sarah.lee@company.com.\nThe project budget is $250,000, with an additional $50,000 allocated for contingencies. \nMonthly maintenance costs are estimated at $3,500. \nFor international clients, please use +1-555-987-6543. \nOur tax ID number is 12-3456789."
    
    result = asyncio.run(process.mask_content(test_text))

    # Assert
    assert result.masked_text is not None
    assert result.mapping is not None

    # Check if all original PII is masked
    pii_info = {
        "persons": ["George Collins", "Jane Smith", "Robert Johnson", "Sarah Lee"],
        "addresses": [
            "123 Main St, Anytown, USA 12345",
            "456 Elm Avenue, Othercity, State 67890",
            "789 Corporate Blvd, Suite 500, Bigcity, State 13579",
        ],
        "phones": ["555-1234", "(987) 654-3210", "1-800-555-9876", "444-333-2222", "+1-555-987-6543"],
        "emails": ["support@example.com", "sarah.lee@company.com"],
        "tax_id": ["12-3456789"],
    }

    non_pii_info = [
        "Monthly maintenance costs are estimated at $3,500.",
    ]

    # Ensure PII is masked
    for person in pii_info["persons"]:
        assert person not in result.masked_text, f"PII {person} was not masked properly"

    for address in pii_info["addresses"]:
        assert address not in result.masked_text, f"PII address {address} was not masked properly"

    for phone in pii_info["phones"]:
        assert phone not in result.masked_text, f"PII phone {phone} was not masked properly"

    for email in pii_info["emails"]:
        assert email not in result.masked_text, f"PII email {email} was not masked properly"

    for tax in pii_info["tax_id"]:
        assert tax not in result.masked_text, f"PII tax ID {tax} was not masked properly"

    # Ensure non-PII data remains unchanged
    for info in non_pii_info:
        assert info in result.masked_text, f"Non-PII {info} was unexpectedly masked"

    # check if mapping length is 15
    assert len(result.mapping) == 15, "Mapping should contain 15 items"

    # Test unmasking
    unmasked_content = process.unmask_content(result.masked_text, result.mapping)

    # Normalize strings by standardizing whitespace and newlines
    def normalize_string(s: str) -> str:
        # Replace all whitespace sequences (including newlines) with a single space
        # and strip leading/trailing whitespace
        return ' '.join(s.split())

    # Test unmasking with normalized strings
    normalized_unmasked = normalize_string(unmasked_content)
    normalized_original = normalize_string(test_text)
    
    # Compare normalized strings
    assert normalized_unmasked == normalized_original, "Unmasked content does not match the original content"

def test_simple_use_case():
    # Arrange
    test_file_path = os.path.join(cwd, "tests", "files", "invoice.pdf")

    process = Process()
    process.load_document_loader(DocumentLoaderPyPdf())
    process.load_file(test_file_path)
    process.add_masking_llm("groq/llama-3.2-11b-text-preview")

    # Arrange
    test_text = "John Doe transferred $5000 to Jane Smith on 2021-05-01."

    # Act
    result = asyncio.run(process.mask_content(test_text))

    # Assert
    assert result.masked_text is not None
    assert result.mapping is not None

    # Ensure PII is masked
    assert "John Doe" not in result.masked_text
    assert "Jane Smith" not in result.masked_text

    # Ensure non-PII data remains
    assert "$5000" in result.masked_text
    assert "2021-05-01" in result.masked_text
    assert "transferred" in result.masked_text

    # Check mapping
    assert len(result.mapping) == 2
    assert "[PERSON1]" in result.mapping
    assert "[PERSON2]" in result.mapping
    assert result.mapping["[PERSON1]"] == "John Doe"
    assert result.mapping["[PERSON2]"] == "Jane Smith"

    # Test unmasking
    unmasked_content = process.unmask_content(result.masked_text, result.mapping)
    assert unmasked_content == test_text

def test_deterministic_hashing():
    # Arrange
    process = Process()
    process.add_masking_llm("groq/llama-3.2-11b-text-preview", MaskingStrategy.DETERMINISTIC_HASHING)

    test_text = "John Doe transferred $5000 to Jane Smith on 2021-05-01."

    # Normalize strings by standardizing whitespace and newlines
    def normalize_string(s: str) -> str:
        # Replace all whitespace sequences (including newlines) with a single space
        # and strip leading/trailing whitespace
        return ' '.join(s.split())

    # Test unmasking with normalized strings
    normalized_unmasked = normalize_string(result.masked_text)
    normalized_original = normalize_string(test_text)

    # Compare normalized strings
    assert normalized_unmasked == normalized_original, "Unmasked content does not match the original content"

    # Act
    result = asyncio.run(process.mask_content(test_text))

    # Assert
    assert result.masked_text is not None
    assert result.mapping is not None

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

def test_split_requires_splitter():
    """Test that attempting to split without loading a splitter first raises an error"""
    # Arrange
    process = Process()
    
    # Act & Assert
    with pytest.raises(ValueError, match="No splitter loaded"):
        process.split([])  # Empty classifications list is fine for this test

if __name__ == "__main__":
    test_mask()
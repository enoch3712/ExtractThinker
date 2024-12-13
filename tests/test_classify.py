import os
import asyncio
from dotenv import load_dotenv
from extract_thinker.document_loader.document_loader_aws_textract import DocumentLoaderAWSTextract
from extract_thinker.document_loader.document_loader_txt import DocumentLoaderTxt
from extract_thinker.extractor import Extractor
from extract_thinker.models.classification_node import ClassificationNode
from extract_thinker.models.classification_tree import ClassificationTree
from extract_thinker.process import Process, ClassificationStrategy
from extract_thinker.document_loader.document_loader_tesseract import DocumentLoaderTesseract
from extract_thinker.models.classification import Classification
from extract_thinker.models.classification_response import ClassificationResponse
from tests.models.invoice import CreditNoteContract, FinancialContract, InvoiceContract
from tests.models.driver_license import DriverLicense, IdentificationContract

# Setup environment and common paths
load_dotenv()
tesseract_path = os.getenv("TESSERACT_PATH")
CURRENT_WORKING_DIRECTORY = os.getcwd()
INVOICE_FILE_PATH = os.path.join(CURRENT_WORKING_DIRECTORY, "tests", "test_images", "invoice.png")
DRIVER_LICENSE_FILE_PATH = os.path.join(CURRENT_WORKING_DIRECTORY, "tests", "test_images", "driver_license.png")

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


def setup_process_with_textract_extractor():
    """Sets up and returns a process configured with only the Textract extractor."""
    # Initialize the Textract document loader
    document_loader = DocumentLoaderAWSTextract()

    # Initialize the Textract extractor
    textract_extractor = Extractor(document_loader)
    textract_extractor.load_llm("gpt-4o")

    # Create the process with only the Textract extractor
    process = Process()
    process.add_classify_extractor([[textract_extractor]])

    return process


def setup_process_with_gpt4_extractor():
    """Sets up and returns a process configured with only the GPT-4 extractor."""
    tesseract_path = os.getenv("TESSERACT_PATH")
    if not tesseract_path:
        raise ValueError("TESSERACT_PATH environment variable is not set")
    print(f"Tesseract path: {tesseract_path}")
    document_loader = DocumentLoaderTesseract(tesseract_path)

    # Initialize the GPT-4 extractor
    gpt_4_extractor = Extractor(document_loader)
    gpt_4_extractor.load_llm("gpt-4o")

    # Create the process with only the GPT-4 extractor
    process = Process()
    process.add_classify_extractor([[gpt_4_extractor]])

    return process


def test_classify_feature():
    """Test classification using a single feature."""
    extractor = setup_extractors()[1]
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
    result = process.classify(INVOICE_FILE_PATH, COMMON_CLASSIFICATIONS, strategy=ClassificationStrategy.CONSENSUS_WITH_THRESHOLD, threshold=9)

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


def test_with_image():
    """Test classification using both consensus and higher order strategies with a threshold."""
    process = setup_process_with_gpt4_extractor()

    COMMON_CLASSIFICATIONS[0].contract = DriverLicense
    COMMON_CLASSIFICATIONS[1].contract = InvoiceContract

    COMMON_CLASSIFICATIONS[0].image = DRIVER_LICENSE_FILE_PATH
    COMMON_CLASSIFICATIONS[1].image = INVOICE_FILE_PATH

    result = process.classify(DRIVER_LICENSE_FILE_PATH, COMMON_CLASSIFICATIONS, strategy=ClassificationStrategy.CONSENSUS, image=True)

    assert result is not None
    assert isinstance(result, ClassificationResponse)
    assert result.name == COMMON_CLASSIFICATIONS[0].name


def test_with_tree():
    """Test classification using the tree strategy"""
    process = setup_process_with_gpt4_extractor()

    financial_docs = ClassificationNode(
        name="Financial Documents",
        classification=Classification(
            name="Financial Documents",
            description="This is a financial document",
            contract=FinancialContract,
        ),
        children=[
            ClassificationNode(
                name="Invoice",
                classification=Classification(
                    name="Invoice",
                    description="This is an invoice",
                    contract=InvoiceContract,
                )
            ),
            ClassificationNode(
                name="Credit Note",
                classification=Classification(
                    name="Credit Note",
                    description="This is a credit note",
                    contract=CreditNoteContract,
                )
            )
        ]
    )

    legal_docs = ClassificationNode(
        name="Identity Documents",
        classification=Classification(
            name="Identity Documents",
            description="This is an identity document",
            contract=IdentificationContract,
        ),
        children=[
            ClassificationNode(
                name="Driver License",
                classification=Classification(
                    name="Driver License",
                    description="This is a driver license",
                    contract=DriverLicense,
                )
            )
        ]
    )

    classification_tree = ClassificationTree(
        nodes=[financial_docs, legal_docs]
    )

    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, 'files','invoice.pdf')

    result = process.classify(pdf_path, classification_tree, threshold=7)

    assert result is not None
    assert result.name == "Invoice"

def test_mom_classification_layers():
    """Test Mixture of Models (MoM) classification with multiple layers."""
    # Arrange
    document_loader = DocumentLoaderTxt()
    
    # Get test file path 
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    CREDIT_NOTE_PATH = os.path.join(CURRENT_DIR, "files", "ambiguous_credit_note.txt")
    
    # Create ambiguous classifications
    test_classifications = [
        Classification(
            name="Receipt",
            description="A document showing payment received for goods or services, typically including items purchased, amounts, and payment method",
            contract=InvoiceContract
        ),
        Classification(
            name="Credit Note",
            description="A document issued to reverse a previous transaction, showing returned items and credit amount, usually referencing an original invoice",
            contract=CreditNoteContract
        )
    ]
    
    # Initialize extractors with different models
    # Layer 1: Small models that might disagree
    gpt35_extractor = Extractor(document_loader)
    gpt35_extractor.load_llm("gpt-3.5-turbo")
    
    claude_haiku_extractor = Extractor(document_loader)
    claude_haiku_extractor.load_llm("claude-3-haiku-20240307")
    
    # Layer 2: More capable models for resolution
    gpt4_extractor = Extractor(document_loader)
    gpt4_extractor.load_llm("gpt-4o")
    sonnet_extractor = Extractor(document_loader)
    sonnet_extractor.load_llm("claude-3-5-sonnet-20241022")
    
    # Create process with multiple layers
    process = Process()
    process.add_classify_extractor([
        [gpt35_extractor, claude_haiku_extractor],  # Layer 1: Small models
        [gpt4_extractor, sonnet_extractor]          # Layer 2: Resolution model
    ])
    
    # Test full MoM process (should resolve using Layer 2)
    final_result = process.classify(
        CREDIT_NOTE_PATH,
        test_classifications,
        strategy=ClassificationStrategy.CONSENSUS_WITH_THRESHOLD,
        threshold=8
    )
    
    # Print results for debugging
    print("\nMoM Classification Results:")
    print(f"Final Classification: {final_result.name}")
    print(f"Confidence: {final_result.confidence}")
    
    # Assertions
    assert final_result is not None, "MoM should produce a result"
    assert final_result.name == "Credit Note", "Final classification should be Credit Note"
    assert final_result.confidence >= 8, "Final confidence should be high"
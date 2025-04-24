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
from extract_thinker.global_models import get_lite_model, get_big_model
import pytest
from pydantic import BaseModel

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

# Dummy contracts for the large tree example
class BankStatementContract(BaseModel): pass
class ContractAgreementContract(BaseModel): pass
class LegalNoticeContract(BaseModel): pass
class PassportContract(BaseModel): pass
class SalesInvoiceContract(BaseModel): pass
class PurchaseInvoiceContract(BaseModel): pass

def setup_extractors():
    """Sets up and returns a list of configured extractors."""
    tesseract_path = os.getenv("TESSERACT_PATH")
    document_loader = DocumentLoaderTesseract(tesseract_path)

    extractors = [
        (get_lite_model(), get_lite_model()),
        (get_big_model(), get_big_model()),
        (get_big_model(), get_big_model())
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

    # Initialize the GPT-4 extractor using the big model
    gpt_4_extractor = Extractor(document_loader)
    gpt_4_extractor.load_llm(get_big_model())

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
    assert result.classification is not None
    assert result.classification.name == "Invoice"
    assert result.classification.description == "This is an invoice"


def test_classify_async():
    """Test asynchronous classification."""
    process = arrange_process_with_extractors()
    result = asyncio.run(process.classify_async(INVOICE_FILE_PATH, COMMON_CLASSIFICATIONS))

    assert result is not None
    assert isinstance(result, ClassificationResponse)
    assert result.name == "Invoice"
    assert result.classification is not None
    assert result.classification.name == "Invoice"
    assert result.classification.description == "This is an invoice"


def test_classify_consensus():
    """Test classification using consensus strategy."""
    process = arrange_process_with_extractors()
    result = process.classify(INVOICE_FILE_PATH, COMMON_CLASSIFICATIONS, strategy=ClassificationStrategy.CONSENSUS)

    assert result is not None
    assert isinstance(result, ClassificationResponse)
    assert result.name == "Invoice"
    assert result.classification is not None
    assert result.classification.name == "Invoice"
    assert result.classification.description == "This is an invoice"


def test_classify_higher_order():
    """Test classification using higher order strategy."""
    process = arrange_process_with_extractors()

    # Act
    result = process.classify(INVOICE_FILE_PATH, COMMON_CLASSIFICATIONS, strategy=ClassificationStrategy.HIGHER_ORDER)

    # Assert
    assert result is not None
    assert isinstance(result, ClassificationResponse)
    assert result.name == "Invoice"
    assert result.classification is not None
    assert result.classification.name == "Invoice"
    assert result.classification.description == "This is an invoice"


def test_classify_both():
    """Test classification using both consensus and higher order strategies with a threshold."""
    process = arrange_process_with_extractors()
    result = process.classify(INVOICE_FILE_PATH, COMMON_CLASSIFICATIONS, strategy=ClassificationStrategy.CONSENSUS_WITH_THRESHOLD, threshold=9)

    assert result is not None
    assert isinstance(result, ClassificationResponse)
    assert result.name == "Invoice"
    assert result.classification is not None
    assert result.classification.name == "Invoice"
    assert result.classification.description == "This is an invoice"


def test_with_contract():
    """Test classification using both consensus and higher order strategies with a threshold."""
    process = arrange_process_with_extractors()

    COMMON_CLASSIFICATIONS[0].contract = InvoiceContract
    COMMON_CLASSIFICATIONS[1].contract = DriverLicense

    result = process.classify(INVOICE_FILE_PATH, COMMON_CLASSIFICATIONS, strategy=ClassificationStrategy.CONSENSUS)

    assert result is not None
    assert isinstance(result, ClassificationResponse)
    assert result.name == "Invoice"
    assert result.classification is not None
    assert result.classification.name == "Invoice"
    assert result.classification.description == "This is an invoice"


def test_with_image():
    """Test classification using image comparison for both driver license and invoice."""
    process = setup_process_with_gpt4_extractor()

    COMMON_CLASSIFICATIONS[0].contract = DriverLicense
    COMMON_CLASSIFICATIONS[1].contract = InvoiceContract

    COMMON_CLASSIFICATIONS[0].image = DRIVER_LICENSE_FILE_PATH
    COMMON_CLASSIFICATIONS[1].image = INVOICE_FILE_PATH

    # Test driver license classification
    result = process.classify(DRIVER_LICENSE_FILE_PATH, COMMON_CLASSIFICATIONS, strategy=ClassificationStrategy.CONSENSUS, image=True)

    assert result is not None
    assert isinstance(result, ClassificationResponse)
    assert result.name == COMMON_CLASSIFICATIONS[0].name
    assert result.classification is not None
    assert result.classification.name == COMMON_CLASSIFICATIONS[0].name
    assert result.classification.description == "This is a driver license"

    # Test invoice classification
    result = process.classify(INVOICE_FILE_PATH, COMMON_CLASSIFICATIONS, strategy=ClassificationStrategy.CONSENSUS, image=True)

    assert result is not None
    assert isinstance(result, ClassificationResponse)
    assert result.name == COMMON_CLASSIFICATIONS[1].name
    assert result.classification is not None
    assert result.classification.name == COMMON_CLASSIFICATIONS[1].name
    assert result.classification.description == "This is an invoice"


def test_with_tree():
    """Test classification using the tree strategy"""
    process = setup_process_with_gpt4_extractor()

    financial_docs = ClassificationNode(
        name="Financial Document",
        classification=Classification(
            name="Financial Document",
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
    assert result.classification is not None
    assert result.classification.name == "Invoice"
    assert result.classification.description == "This is an invoice"
    assert result.classification.contract == InvoiceContract
    # Verify UUID matching worked (assuming financial_docs.children[0] is the Invoice node)
    expected_invoice_node = next(node for node in financial_docs.children if node.name == "Invoice")
    assert result.classification.uuid == expected_invoice_node.classification.uuid


def test_tree_classification_low_confidence():
    """Test tree classification raises error when confidence is below threshold."""
    process = setup_process_with_gpt4_extractor()

    # Reuse the same tree structure from test_with_tree
    financial_docs = ClassificationNode(
        name="Financial Document",
        classification=Classification(
            name="Financial Document",
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

    # Set an impossibly high threshold
    high_threshold = 10.1 # Force confidence < threshold

    # Assert that the correct ValueError is raised due to low confidence
    with pytest.raises(ValueError):
        process.classify(pdf_path, classification_tree, threshold=high_threshold)


def test_large_classification_tree():
    """Test classification with a larger, multi-level tree."""
    process = setup_process_with_gpt4_extractor()

    # --- Define Tree Structure ---

    # Level 3: Invoice Types
    sales_invoice_node = ClassificationNode(
        name="Sales Invoice",
        classification=Classification(
            name="Sales Invoice",
            description="An invoice sent to a customer detailing products/services sold and amount due.",
            contract=SalesInvoiceContract, # Specific contract for Sales Invoice
        )
    )
    purchase_invoice_node = ClassificationNode(
        name="Purchase Invoice",
        classification=Classification(
            name="Purchase Invoice",
            description="An invoice received from a supplier detailing products/services bought.",
            contract=PurchaseInvoiceContract, # Specific contract for Purchase Invoice
        )
    )

    # Level 2: Financial Subtypes (Invoice node becomes a parent)
    invoice_node = ClassificationNode(
        name="Invoice/Bill", # More general name
        classification=Classification(
            name="Invoice/Bill",
            description="A general bill requesting payment for goods or services.",
            contract=InvoiceContract, # General invoice contract remains here
        ),
        children=[sales_invoice_node, purchase_invoice_node] # Add Level 3 children
    )
    credit_note_node = ClassificationNode(
        name="Credit Note",
        classification=Classification(
            name="Credit Note",
            description="A document correcting a previous invoice or returning funds.",
            contract=CreditNoteContract,
        )
    )
    bank_statement_node = ClassificationNode(
        name="Bank Statement",
        classification=Classification(
            name="Bank Statement",
            description="A summary of financial transactions occurring over a given period.",
            contract=BankStatementContract,
        )
    )

    # Level 1: Financial Documents Root
    financial_docs = ClassificationNode(
        name="Financial Document",
        classification=Classification(
            name="Financial Document",
            description="Documents related to financial transactions or status.",
            contract=FinancialContract,
        ),
        children=[invoice_node, credit_note_node, bank_statement_node]
    )

    # Level 2: Legal Subtypes
    contract_agreement_node = ClassificationNode(
        name="Contract/Agreement",
        classification=Classification(
            name="Contract/Agreement",
            description="A legally binding agreement between parties.",
            contract=ContractAgreementContract,
        )
    )
    legal_notice_node = ClassificationNode(
        name="Legal Notice",
        classification=Classification(
            name="Legal Notice",
            description="A formal notification required or permitted by law.",
            contract=LegalNoticeContract,
        )
    )

    # Level 1: Legal Documents Root
    legal_docs = ClassificationNode(
        name="Legal Document",
        classification=Classification(
            name="Legal Document",
            description="Documents with legal significance or implications.",
            contract=None, # Example: Root might not have a specific contract
        ),
        children=[contract_agreement_node, legal_notice_node]
    )

    # Level 2: Identification Subtypes
    driver_license_node = ClassificationNode(
        name="Driver License",
        classification=Classification(
            name="Driver License",
            description="Official document permitting an individual to operate a motor vehicle.",
            contract=DriverLicense,
        )
    )
    passport_node = ClassificationNode(
        name="Passport",
        classification=Classification(
            name="Passport",
            description="Official government document certifying identity and citizenship for travel.",
            contract=PassportContract,
        )
    )

    # Level 1: Identity Documents Root
    identity_docs = ClassificationNode(
        name="Identification Document",
        classification=Classification(
            name="Identification Document",
            description="Documents used to verify a person's identity.",
            contract=IdentificationContract,
        ),
        children=[driver_license_node, passport_node]
    )

    # Top Level Tree
    classification_tree = ClassificationTree(
        nodes=[financial_docs, legal_docs, identity_docs]
    )

    # --- Perform Classification ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Using the same invoice PDF as before
    pdf_path = os.path.join(current_dir, 'files', 'invoice.pdf')

    # Classify with a reasonable threshold
    result = process.classify(pdf_path, classification_tree, threshold=7)

    # --- Assert Results ---
    assert result is not None, "Classification should return a result."
    assert result.name == "Sales Invoice", "The document should be classified as a Sales Invoice."
    assert result.classification is not None, "Result should contain classification details."
    # Verify it picked the correct node using UUID
    assert result.classification.uuid == sales_invoice_node.classification.uuid, "Result UUID should match the Sales Invoice node UUID."
    assert result.classification.contract == SalesInvoiceContract, "Result contract should be SalesInvoiceContract."


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
    gpt35_extractor.load_llm(get_big_model())
    
    claude_haiku_extractor = Extractor(document_loader)
    claude_haiku_extractor.load_llm(get_lite_model())
    
    # Layer 2: More capable models for resolution
    gpt4_extractor = Extractor(document_loader)
    gpt4_extractor.load_llm(get_big_model())
    sonnet_extractor = Extractor(document_loader)
    sonnet_extractor.load_llm(get_big_model())
    
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
    assert final_result.classification is not None
    assert final_result.classification.name == "Credit Note"
    assert final_result.classification.description == "A document issued to reverse a previous transaction, showing returned items and credit amount, usually referencing an original invoice"
    assert final_result.classification.contract == CreditNoteContract


if __name__ == "__main__":
    test_tree_classification_low_confidence()
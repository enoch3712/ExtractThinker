import os
from dotenv import load_dotenv
from extract_thinker import Process
from extract_thinker.document_loader.document_loader_pypdf import DocumentLoaderPyPdf
from extract_thinker.extractor import Extractor
from extract_thinker.models.classification import Classification
from extract_thinker.models.contract import Contract

class DriverLicense(Contract):
    name: str
    age: int
    license_number: str

class InvoiceContract(Contract):
    invoice_number: str
    invoice_date: str

load_dotenv()
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INVOICE_PATH = f"{CURRENT_DIR}/../files/invoice.pdf"

def test_critical_classification():
    """Critical test for basic classification"""
    # Setup
    document_loader = DocumentLoaderPyPdf()
    extractor = Extractor(document_loader)
    extractor.load_llm("groq/llama-3.3-70b-versatile")

    process = Process()
    process.add_classify_extractor([[extractor]])

    classifications = [
        Classification(
            name="Invoice",
            description="This is an invoice document", 
            contract=InvoiceContract
        ),
        Classification(
            name="Driver License",
            description="This is a driver license document",
            contract=DriverLicense
        )
    ]

    # Act
    result = process.classify(INVOICE_PATH, classifications)

    # Assert
    assert result is not None
    assert result.name == "Invoice"


if __name__ == "__main__":
    test_critical_classification()
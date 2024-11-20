import os
from dotenv import load_dotenv
from extract_thinker import Extractor
from extract_thinker.document_loader.document_loader_pypdf import DocumentLoaderPyPdf
from tests.models.invoice import InvoiceContract

load_dotenv()
cwd = os.getcwd()

def test_critical_extract_with_pypdf():
    """Critical test for basic extraction functionality"""
    test_file_path = os.path.join(cwd, "tests", "files", "invoice.pdf")

    extractor = Extractor()
    extractor.load_document_loader(DocumentLoaderPyPdf())
    extractor.load_llm("groq/llama-3.1-70b-versatile")

    result = extractor.extract(test_file_path, InvoiceContract)

    assert result is not None
    assert result.lines[0].description == "Consultation services"
    assert result.lines[0].quantity == 3
    assert result.lines[0].unit_price == 375
    assert result.lines[0].amount == 1125
import os
from dotenv import load_dotenv
from extract_thinker.document_loader.document_loader_pypdf import DocumentLoaderPyPdf
from extract_thinker.extractor import Extractor
from tests.models.invoice import InvoiceContract

load_dotenv()
cwd = os.getcwd()


def test_extract_with_ollama():
    test_file_path = os.path.join(cwd, "tests", "files", "invoice.pdf")

    extractor = Extractor()
    extractor.load_document_loader(
        DocumentLoaderPyPdf()
    )

    os.environ["API_BASE"] = "http://localhost:11434"
    extractor.load_llm("ollama/phi3.5")

    # Act
    result = extractor.extract(test_file_path, InvoiceContract)

    # Assert
    assert result is not None
    assert result.invoice_number == "00012"
    assert result.invoice_date == "1/30/23"

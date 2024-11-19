import os
import io
import pytest
from dotenv import load_dotenv

from extract_thinker.document_loader.document_loader_pdfplumber import DocumentLoaderPdfPlumber

load_dotenv()

def test_load_content_from_pdf():
    # Arrange
    loader = DocumentLoaderPdfPlumber()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, 'files', 'invoice.pdf')

    # Act
    result = loader.load_content_from_file(pdf_path)

    # Assert
    assert isinstance(result, dict)
    assert "text" in result
    assert "tables" in result
    assert isinstance(result["text"], list)
    assert isinstance(result["tables"], list)
    assert len(result["text"]) > 0
    assert len(result["tables"]) > 0  # Assuming the invoice has at least one table

if __name__ == "__main__":
    test_load_content_from_pdf()
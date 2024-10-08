import os
import pytest
from dotenv import load_dotenv

from extract_thinker.document_loader.document_loader_aws_textract import DocumentLoaderAWSTextract

load_dotenv()

def test_load_content_from_pdf():
    # Arrange
    loader = DocumentLoaderAWSTextract(
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_DEFAULT_REGION')
    )

    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, 'files','invoice.pdf')

    # Act
    result = loader.load_content_from_file(pdf_path)

    # Assert
    assert isinstance(result, dict)
    assert "pages" in result
    assert "tables" in result
    assert "forms" in result
    assert "layout" in result
    assert len(result["pages"]) > 0
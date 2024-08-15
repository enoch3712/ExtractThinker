import os
import pytest
from moto import mock_textract
import boto3
from dotenv import load_dotenv

from extract_thinker.document_loader.document_loader_aws_textract import DocumentLoaderAWSTextract

load_dotenv()

@pytest.fixture
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    os.environ['AWS_ACCESS_KEY_ID'] = 'testing'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'testing'
    os.environ['AWS_SECURITY_TOKEN'] = 'testing'
    os.environ['AWS_SESSION_TOKEN'] = 'testing'
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

@pytest.fixture
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    return {
        'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
        'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
        'region_name': os.getenv('AWS_DEFAULT_REGION')
    }


def test_load_content_from_pdf(textract_client):
    # Arrange
    loader = DocumentLoaderAWSTextract.from_client(textract_client)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, 'test_files', 'sample.pdf')

    # Act
    result = loader.load_content_from_file(pdf_path)

    # Assert
    assert isinstance(result, dict)
    assert "pages" in result
    assert "tables" in result
    assert "forms" in result
    assert "layout" in result
    assert len(result["pages"]) > 0
    
    # You may want to add more specific
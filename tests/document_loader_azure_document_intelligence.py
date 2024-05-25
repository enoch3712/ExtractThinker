import os

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from io import BytesIO
from dotenv import load_dotenv
import pytest
from azure.core.exceptions import AzureError

from extract_thinker.document_loader.document_loader_azure_document_intelligence import DocumentLoaderAzureForm

cwd = os.getcwd()
load_dotenv()

# Arrange
subscription_key = os.getenv("AZURE_SUBSCRIPTION_KEY")
endpoint = os.getenv("AZURE_ENDPOINT")
loader = DocumentLoaderAzureForm(subscription_key, endpoint)
test_file_path = os.path.join(cwd, "tests", "test_documents", "invoice.pdf")


def test_load_content_from_file():
    # Act
    try:
        content = loader.load_content_from_file("C:\\Users\\Lopez\\Downloads\\LNKD_INVOICE_7894414780.pdf")
    except AzureError as e:
        pytest.fail(f"AzureError occurred: {e}")

    # Assert
    assert content is not None
    assert isinstance(content, list)
    assert len(content) > 0


def test_load_content_from_stream():
    with open(test_file_path, 'rb') as f:
        test_document_stream = BytesIO(f.read())

    # Act
    try:
        content = loader.load_content_from_stream(test_document_stream)
    except AzureError as e:
        pytest.fail(f"AzureError occurred: {e}")

    # Assert
    assert content is not None
    assert isinstance(content, list)
    assert len(content) > 0


def test_cache_for_file():
    # Act
    try:
        content1 = loader.load_content_from_file(test_file_path)
        content2 = loader.load_content_from_file(test_file_path)
    except AzureError as e:
        pytest.fail(f"AzureError occurred: {e}")

    # Assert
    assert content1 is content2


test_load_content_from_file()
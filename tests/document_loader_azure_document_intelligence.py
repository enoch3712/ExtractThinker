import os
from dotenv import load_dotenv

from extract_thinker.document_loader.document_loader_azure_document_intelligence import DocumentLoaderAzureForm

cwd = os.getcwd()
load_dotenv()

# Arrange
subscription_key = os.getenv("AZURE_SUBSCRIPTION_KEY")
endpoint = os.getenv("AZURE_ENDPOINT")
loader = DocumentLoaderAzureForm(subscription_key, endpoint)
test_file_path = os.path.join(cwd, "test_images", "invoice.png")


def test_load_content_from_file():
    # Act
    content = loader.load_content_from_file(test_file_path)

    firstPage = content["pages"][0]

    # Assert
    assert firstPage is not None
    assert firstPage["paragraphs"][0] == "Invoice 0000001"
    assert len(firstPage["tables"][0]) == 4

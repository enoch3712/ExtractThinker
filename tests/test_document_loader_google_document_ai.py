import os
from dotenv import load_dotenv

from extract_thinker.document_loader.document_loader_google_document_ai import (
    DocumentLoaderDocumentAI,
)

cwd = os.getcwd()
load_dotenv()

google_credentials = os.getenv("DOCUMENTAI_GOOGLE_CREDENTIALS")
processor_name = os.getenv("DOCUMENTAI_PROCESSOR_NAME")
location = os.getenv("DOCUMENTAI_LOCATION")
loader = DocumentLoaderDocumentAI(
    credentials=google_credentials, location=location, processor_name=processor_name
)


def test_load_content_from_cv_file():
    test_file_path = os.path.join(cwd, "tests", "files", "CV_Candidate.pdf")
    content = loader.load_content_from_file(test_file_path)

    firstPage = content["pages"][0]

    assert firstPage is not None

    assert len(firstPage["paragraphs"]) > 0
    assert firstPage["tables"] == []

    assert "johndoe@example.com" in firstPage["content"]
    assert "React Professional Certification" in firstPage["content"]


def test_load_content_from_form_with_table_file():
    test_file_path = os.path.join(cwd, "tests", "files", "form_with_tables.pdf")
    content = loader.load_content_from_file(test_file_path)

    firstPage = content["pages"][0]
    assert firstPage is not None
    assert len(firstPage["tables"]) == 1
    assert ",".join(firstPage["tables"][0][0]) == "Item,Description"
    assert ",".join(firstPage["tables"][0][3]) == "Item 3,Description 3"

    assert "12345678" in firstPage["content"]
    assert "123 Fake St" in firstPage["content"]

def test_load_content_file_as_stream():
    test_file_path = os.path.join(cwd, "tests", "files", "CV_Candidate.pdf")
    with open(test_file_path, "rb") as f:
        content = loader.load_content_from_stream(stream=f, mime_type="application/pdf")

    assert content is not None
    assert len(content["pages"]) > 0

    firstPage = content["pages"][0]
    assert "johndoe@example.com" in firstPage["content"]
    assert "React Professional Certification" in firstPage["content"]

def test_load_content_from_file_vision_mode():
    # Arrange
    loader = DocumentLoaderDocumentAI(
        credentials=google_credentials,
        location=location,
        processor_name=processor_name
    )
    loader.set_vision_mode(True)
    test_file_path = os.path.join(cwd, "tests", "files", "CV_Candidate.pdf")

    # Act
    result = loader.load(test_file_path)

    # Assert
    assert isinstance(result, dict)
    assert "images" in result
    assert len(result["images"]) > 0
    # Verify each image is bytes
    for page_num, image_data in result["images"].items():
        assert isinstance(page_num, int)
        assert isinstance(image_data, bytes)
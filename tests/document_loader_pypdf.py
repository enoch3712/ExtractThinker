import os
from dotenv import load_dotenv
from extract_thinker.document_loader.document_loader_pypdf import DocumentLoaderPyPdf

cwd = os.getcwd()
load_dotenv()

# Arrange
loader = DocumentLoaderPyPdf()
test_file_path = os.path.join(cwd, "files", "CV_Candidate.pdf")


def test_load_content_from_file():
    # Act
    content = loader.load_content_from_file(test_file_path)

    # Convert the list of words into a single string
    content_text = " ".join(content["text"])

    # Assert
    assert content is not None
    assert "University of New York" in content_text
    assert "XYZ Innovations" in content_text

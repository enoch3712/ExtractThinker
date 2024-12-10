import os
from extract_thinker.document_loader.document_loader_txt import DocumentLoaderTxt

def test_load_content_from_file():
    # Arrange
    loader = DocumentLoaderTxt()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(current_dir, 'files', 'ambiguous_credit_note.txt')
    
    # Act
    result = loader.load_content_from_file(txt_path)
    
    # Assert
    assert isinstance(result, str)
    assert "CREDIT NOTE / RECEIPT" in result
    assert "CN-2024-001" in result
    assert "John Smith" in result
    assert "Total Credit        $137.50" in result 
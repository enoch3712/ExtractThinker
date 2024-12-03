import os
import io
from extract_thinker.document_loader.document_loader_doc2txt import DocumentLoaderDoc2txt

def test_load_content_from_word():
    # Arrange
    loader = DocumentLoaderDoc2txt()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    word_path = os.path.join(current_dir, 'files', 'invoice.docx')

    # Act
    result = loader.load_content_from_file(word_path)

    # Assert
    assert isinstance(result, str)
    assert len(result) > 0

def test_load_content_from_word_as_list():
    # Arrange
    loader = DocumentLoaderDoc2txt()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    word_path = os.path.join(current_dir, 'files', 'invoice.docx')

    # Act
    result = loader.load_content_from_file_list(word_path)

    # Assert
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(item, str) for item in result)

def test_load_content_from_stream():
    # Arrange
    loader = DocumentLoaderDoc2txt()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    word_path = os.path.join(current_dir, 'files', 'invoice.docx')
    
    with open(word_path, 'rb') as file:
        stream = io.BytesIO(file.read())
        
        # Act
        result = loader.load_content_from_stream(stream)

        # Assert
        assert isinstance(result, str)
        assert len(result) > 0
        
def test_vision_mode_error():
    loader = DocumentLoaderDoc2txt()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    word_path = os.path.join(current_dir, 'files', 'invoice.docx')
    loader.set_vision_mode(True)
    
    try: 
        loader.load(word_path)
    except ValueError as e:
        assert 'Source cannot be processed in vision mode. Only PDFs and images are supported.' in str(e)
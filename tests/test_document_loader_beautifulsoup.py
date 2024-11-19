import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from extract_thinker.document_loader.beautiful_soup_web_loader import DocumentLoaderBeautifulSoup

def test_load_content_from_google():
    # Arrange
    loader = DocumentLoaderBeautifulSoup(header_handling="summarize")
    url = "https://www.google.com"
    
    # Act
    content = loader.load_content_from_file(url)
    
    # Assert
    assert isinstance(content, dict)
    assert "content" in content
    assert "Google" in content["content"]

if __name__ == "__main__":
    test_load_content_from_google()
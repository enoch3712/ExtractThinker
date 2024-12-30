# Web Document Loader

The Web loader in ExtractThinker uses BeautifulSoup to extract content from web pages and HTML documents.

## Basic Usage

Here's how to use the Web loader:

```python
from extract_thinker import Extractor
from extract_thinker.document_loader import DocumentLoaderBeautifulSoup

# Initialize the loader
loader = DocumentLoaderBeautifulSoup(
    header_handling="summarize"  # Options: summarize, extract, ignore
)

# Load content from URL
content = loader.load_content_from_file("https://example.com")

# Access extracted content
text = content["content"]
```

## Features

- HTML content extraction
- Header/footer handling
- Link extraction

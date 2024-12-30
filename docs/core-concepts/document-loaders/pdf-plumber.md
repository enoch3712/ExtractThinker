# PDF Plumber Document Loader

PDF Plumber is a Python library for extracting text and tables from PDFs. ExtractThinker's PDF Plumber loader provides a simple interface for working with this library.

## Basic Usage

Here's how to use the PDF Plumber loader:

```python
from extract_thinker import Extractor
from extract_thinker.document_loader import DocumentLoaderPdfPlumber

# Initialize the loader
loader = DocumentLoaderPdfPlumber()

# Load document content
result = loader.load_content_from_file("document.pdf")

# Access extracted content
text = result["text"]      # List of text content by page
tables = result["tables"]  # List of tables found in document
```

## Features

- Text extraction with positioning
- Table detection and extraction
- Image location detection
- Character-level text properties
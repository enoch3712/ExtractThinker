# PyPDF Document Loader

PyPDF is a pure-Python library for reading and writing PDFs. ExtractThinker's PyPDF loader provides a simple interface for text extraction.

## Basic Usage

Here's how to use the PyPDF loader:

```python
from extract_thinker import Extractor
from extract_thinker.document_loader import DocumentLoaderPyPdf

# Initialize the loader
loader = DocumentLoaderPyPdf()

# Load document content
content = loader.load_content_from_file("document.pdf")

# Access text content
text = content["text"]  # List of text content by page
```

## Features

- Basic text extraction
- Page-by-page processing
- Metadata extraction
- Low memory footprint
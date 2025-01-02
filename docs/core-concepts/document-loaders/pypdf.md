# PyPDF Document Loader

PyPDF is a pure-Python library for reading and writing PDFs. ExtractThinker's PyPDF loader provides a simple interface for text extraction.

## Installation

Install the required dependencies:

```bash
pip install pypdf
```

## Supported Formats

- pdf

## Usage

```python
from extract_thinker import DocumentLoaderPyPdf

# Initialize the loader
loader = DocumentLoaderPyPdf()

# Load document
pages = loader.load("path/to/your/document.pdf")

# Process extracted content
for page in pages:
    # Access text content
    text = page["content"]
```

## Features

- Basic text extraction from PDFs
- Support for image extraction in vision mode
- Lightweight and fast processing
- Memory efficient for large PDFs
# PDFPlumber Document Loader

The PDFPlumber loader uses the pdfplumber library to extract text and tables from PDF documents with high accuracy.

## Installation

Install the required dependencies:

```bash
pip install pdfplumber
```

## Supported Formats

- `PDF`

## Usage

```python
from extract_thinker import DocumentLoaderPdfPlumber

# Initialize the loader
loader = DocumentLoaderPdfPlumber()

# Load document
pages = loader.load("path/to/your/document.pdf")

# Process extracted content
for page in pages:
    # Access text content
    text = page["content"]
    
    # Access tables (if any)
    tables = page["tables"]
```

## Features

- Text extraction with layout preservation
- Table detection and extraction with multiple strategies
- Automatic table cleaning and formatting
- Handles complex PDF layouts

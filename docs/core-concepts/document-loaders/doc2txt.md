# Microsoft Word Document Loader (Doc2txt)

The Doc2txt loader is designed to handle Microsoft Word documents (`.doc` and `.docx` files). It uses the `docx2txt` library to extract text content from Word documents.

## Installation

Install the required dependencies:

```bash
pip install docx2txt
```

## Supported Formats

- doc
- docx

## Usage

```python
from extract_thinker import DocumentLoaderDoc2txt

# Initialize the loader
loader = DocumentLoaderDoc2txt()

# Load document
pages = loader.load("path/to/your/document.docx")

# Process extracted content
for page in pages:
    # Access text content
    text = page["content"]
```

## Features

- Text extraction from Word documents
- Support for both .doc and .docx formats
- Automatic page detection
- Preserves basic text formatting
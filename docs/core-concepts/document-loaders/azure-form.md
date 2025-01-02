# Azure Document Intelligence Document Loader

The Azure Document Intelligence loader (formerly known as Form Recognizer) uses Azure's Document Intelligence service to extract text, tables, and layout information from documents.

## Installation

Install the required dependencies:

```bash
pip install azure-ai-formrecognizer
```

## Prerequisites

1. An Azure subscription
2. A Document Intelligence resource created in your Azure portal
3. The endpoint URL and subscription key from your Azure resource

## Supported Formats

Supports `PDF`, `JPEG/JPG`, `PNG`, `BMP`, `TIFF`, `HEIF`, `DOCX`, `XLSX`, `PPTX` and `HTML`.

## Usage

```python
from extract_thinker import DocumentLoaderAzureForm

# Initialize the loader
loader = DocumentLoaderAzureForm(
    subscription_key="your-subscription-key",
    endpoint="your-endpoint-url"
)

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
- Table detection and extraction
- Support for multiple document formats
- Automatic table content deduplication from text

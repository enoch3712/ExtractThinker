# Google Document AI Document Loader

The Google Document AI loader uses Google Cloud's Document AI service to extract text, tables, forms, and key-value pairs from documents.

## Installation

You need Google Cloud credentials and a Document AI processor. You will need:
- `DOCUMENTAI_GOOGLE_CREDENTIALS`
- `DOCUMENTAI_LOCATION`
- `DOCUMENTAI_PROCESSOR_NAME`

```bash
pip install google-cloud-documentai google-api-core google-oauth2-tool
```

## Basic Usage

Here's a simple example of using the Google Document AI loader:

```python
from extract_thinker import DocumentLoaderGoogleDocumentAI

# Initialize the loader
loader = DocumentLoaderGoogleDocumentAI(
    project_id="your-project-id",
    location="us",  # or "eu"
    processor_id="your-processor-id",
    credentials="path/to/service-account.json"  # or JSON string
)

# Load document
pages = loader.load("path/to/your/document.pdf")

# Process extracted content
for page in pages:
    # Access text content
    text = page["content"]
    
    # Access tables (if any)
    tables = page["tables"]
    
    # Access form fields (if any)
    forms = page["forms"]
    
    # Access key-value pairs (if any)
    key_values = page["key_value_pairs"]
```

## Features

Document AI supports `PDF`, `TIFF`, `GIF`, `JPEG`, `PNG` with a maximum file size of 20MB or 2000 pages.

- Text extraction with layout preservation
- Table detection and extraction
- Form field detection
- Key-value pair extraction
- Support for multiple document formats

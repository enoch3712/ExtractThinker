# Azure Document Intelligence

> Azure Document Intelligence (formerly known as `Azure Form Recognizer`) is a machine-learning based service that extracts texts (including handwriting), tables, document structures (e.g., titles, section headings, etc.) and key-value-pairs from digital or scanned PDFs, images, Office and HTML files.

## Prerequisite

An Azure Document Intelligence resource in one of the 3 preview regions: `East US`, `West US2`, `West Europe`. You will be passing `<endpoint>` and `<key>` as parameters to the loader.

```python
%pip install --upgrade --quiet extract_thinker azure-ai-formrecognizer
```

## Basic Usage

Here's a simple example of using the Azure Document Intelligence Loader:

```python
from extract_thinker import Extractor
from extract_thinker.document_loader import DocumentLoaderAzureForm

# Initialize the loader with Azure credentials
subscription_key = os.getenv("AZURE_SUBSCRIPTION_KEY")
endpoint = os.getenv("AZURE_ENDPOINT")

loader = DocumentLoaderAzureForm(subscription_key, endpoint)

# Load document
content = loader.load("invoice.pdf")

# Get content list (page by page)
content_list = loader.load_content_list("invoice.pdf")
```

## Advanced Configuration

The loader provides advanced features for handling tables and document structure:

```python
# The result will contain:
# - Paragraphs (text content)
# - Tables (structured data)
# Each page is processed separately

result = loader.load("document.pdf")
for page in result["pages"]:
    # Access paragraphs
    for paragraph in page["paragraphs"]:
        print(f"Text: {paragraph}")
    
    # Access tables
    for table in page["tables"]:
        print(f"Table data: {table}")
```

Document Intelligence supports `PDF`, `JPEG/JPG`, `PNG`, `BMP`, `TIFF`, `HEIF`, `DOCX`, `XLSX`, `PPTX` and `HTML`.

## Best Practices

- Use high-quality scans for best results
- Consider caching results (built-in TTL of 300 seconds)
- Handle tables and paragraphs separately for better accuracy
- Process documents page by page for large files

For more examples and advanced usage, check out the [examples directory](examples/) in the repository. 
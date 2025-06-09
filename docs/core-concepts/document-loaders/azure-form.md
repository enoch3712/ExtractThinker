# Azure Document Intelligence Loader

The Azure Document Intelligence loader (formerly Form Recognizer) uses Azure's Document Intelligence service to extract text, tables, and structured information from documents.

## Supported Formats

Supports `PDF`, `JPEG/JPG`, `PNG`, `BMP`, `TIFF`, `HEIF`, `DOCX`, `XLSX`, `PPTX` and `HTML`.

## Usage

### Basic Usage

```python
from extract_thinker import DocumentLoaderAzureForm

# Initialize with Azure credentials
loader = DocumentLoaderAzureForm(
    subscription_key="your_subscription_key",
    endpoint="your_endpoint",
    model_id="prebuilt-document"  # Use prebuilt document model
)

# Load document
pages = loader.load("path/to/your/document.pdf")

# Process extracted content
for page in pages:
    # Access text content
    text = page["content"]
    # Access tables if available
    tables = page.get("tables", [])
    # Access form fields if available
    forms = page.get("forms", {})
```

### Configuration-based Usage

```python
from extract_thinker import DocumentLoaderAzureForm, AzureConfig

# Create configuration
config = AzureConfig(
    subscription_key="your_subscription_key",
    endpoint="your_endpoint",
    model_id="prebuilt-layout",     # Use layout model for enhanced layout analysis
    cache_ttl=600,                  # Cache results for 10 minutes
    features=["ocrHighResolution", "barcodes"]  # Enable advanced features
)

# Initialize loader with configuration
loader = DocumentLoaderAzureForm(config)

# Load and process document
pages = loader.load("path/to/your/document.pdf")
```

### Advanced Features Usage

```python
from extract_thinker import DocumentLoaderAzureForm, AzureConfig

# Configuration with multiple advanced features
config = AzureConfig(
    subscription_key="your_subscription_key",
    endpoint="your_endpoint",
    model_id="prebuilt-layout",
    features=[
        "ocrHighResolution",    # High resolution OCR for small text
        "formulas",             # Extract mathematical formulas in LaTeX
        "styleFont",            # Extract font properties
        "barcodes",             # Extract barcodes and QR codes
        "languages",            # Detect document languages
        "keyValuePairs"         # Extract key-value pairs from forms
    ]
)

loader = DocumentLoaderAzureForm(config)
pages = loader.load("document_with_advanced_content.pdf")

for page in pages:
    # Standard content
    print(f"Text content: {page['content']}")
    print(f"Tables: {page['tables']}")
    print(f"Forms: {page['forms']}")
    
    # Advanced features (if detected in document)
    if 'formulas' in page:
        print(f"Mathematical formulas: {page['formulas']}")
    
    if 'fonts' in page:
        print(f"Font information: {page['fonts']}")
    
    if 'barcodes' in page:
        print(f"Barcodes found: {page['barcodes']}")
    
    if 'languages' in page:
        print(f"Detected languages: {page['languages']}")
```

### Specialized Models Usage

```python
# Use specialized invoice model
config = AzureConfig(
    subscription_key="your_subscription_key",
    endpoint="your_endpoint",
    model_id="prebuilt-invoice"
)

loader = DocumentLoaderAzureForm(config)
pages = loader.load("invoice.pdf")

# Access extracted invoice fields
for page in pages:
    forms = page["forms"]
    vendor_name = forms.get("VendorName", "")
    invoice_total = forms.get("InvoiceTotal", "")
    print(f"Vendor: {vendor_name}, Total: {invoice_total}")
```

## Configuration Options

The `AzureConfig` class supports the following options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `subscription_key` | str | Required | Azure subscription key |
| `endpoint` | str | Required | Azure endpoint URL |
| `content` | Any | None | Initial content to process |
| `cache_ttl` | int | 300 | Cache time-to-live in seconds |
| `model_id` | str | "prebuilt-layout" | Model ID to use |
| `max_retries` | int | 3 | Maximum retries for failed requests |
| `features` | List[str] | None | Advanced features to enable |

## Available Models

### General Purpose Models

| Model ID | Description | Best For |
|----------|-------------|----------|
| `prebuilt-read` | OCR/Read model | Text extraction from printed and handwritten documents |
| `prebuilt-layout` | Layout analysis | Documents with tables, selection marks, and complex layouts |
| `prebuilt-document` | General document | Key-value pairs, tables, and general document structure |

### Specialized Models

| Model ID | Description |
|----------|-------------|
| `prebuilt-invoice` | Invoice processing |
| `prebuilt-receipt` | Receipt processing |
| `prebuilt-idDocument` | Identity documents |
| `prebuilt-businessCard` | Business cards |
| `prebuilt-tax.us.w2` | US W2 tax forms |
| `prebuilt-tax.us.1040` | US 1040 tax forms |
| `prebuilt-contract` | Contracts |
| `prebuilt-healthInsurance` | US health insurance cards |
| `prebuilt-bankStatement` | Bank statements |
| `prebuilt-payStub` | Pay stubs |

## Advanced Features

The loader supports advanced extraction features that can be enabled via the `features` parameter:

| Feature | Description | Output Field |
|---------|-------------|--------------|
| `ocrHighResolution` | High resolution OCR for better small text recognition | Enhanced text in `content` |
| `formulas` | Extract mathematical formulas in LaTeX format | `formulas` array |
| `styleFont` | Extract font properties (family, style, weight, color) | `fonts` array |
| `barcodes` | Extract barcodes and QR codes | `barcodes` array |
| `languages` | Detect document languages | `languages` array |
| `keyValuePairs` | Extract key-value pairs from forms | Enhanced `forms` dict |
| `queryFields` | Enable custom field extraction | Enhanced extraction |
| `searchablePDF` | Convert scanned PDFs to searchable format | Enhanced OCR |

## Features

- Text extraction with layout preservation
- Table detection and extraction
- Form field recognition with specialized models
- Advanced OCR with high resolution support
- Mathematical formula extraction (LaTeX format)
- Font property extraction
- Barcode and QR code detection
- Multi-language document support
- Caching support with configurable TTL
- Vision mode support for image formats
- Retry logic for robust processing

## Notes

- Azure subscription key and endpoint are required
- Advanced features may increase processing time and costs
- Specialized models are optimized for specific document types
- Rate limits and quotas apply based on your Azure subscription
- Vision mode is supported for image formats
- High resolution OCR is recommended for documents with small text
- Formula extraction works best with clear mathematical notation

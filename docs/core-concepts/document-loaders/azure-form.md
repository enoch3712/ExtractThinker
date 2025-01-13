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
    endpoint="your_endpoint",
    key="your_api_key",
    model="prebuilt-document"  # Use prebuilt document model
)

# Load document
pages = loader.load("path/to/your/document.pdf")

# Process extracted content
for page in pages:
    # Access text content
    text = page["content"]
    # Access tables if available
    tables = page.get("tables", [])
```

### Configuration-based Usage

```python
from extract_thinker import DocumentLoaderAzureForm, AzureConfig

# Create configuration
config = AzureConfig(
    endpoint="your_endpoint",
    key="your_api_key",
    model="prebuilt-read",     # Use layout model for enhanced layout analysis
    language="en",               # Specify document language
    pages=[1, 2, 3],            # Process specific pages
    cache_ttl=600               # Cache results for 10 minutes
)

# Initialize loader with configuration
loader = DocumentLoaderAzureForm(config)

# Load and process document
pages = loader.load("path/to/your/document.pdf")
```

## Configuration Options

The `AzureConfig` class supports the following options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `content` | Any | None | Initial content to process |
| `cache_ttl` | int | 300 | Cache time-to-live in seconds |
| `endpoint` | str | None | Azure endpoint URL |
| `key` | str | None | Azure API key |
| `model` | str | "prebuilt-document" | Model ID to use |
| `language` | str | None | Document language code |
| `pages` | List[int] | None | Specific pages to process |
| `reading_order` | str | "natural" | Text reading order |

## Features

- Text extraction with layout preservation
- Table detection and extraction
- Form field recognition
- Multiple model support (document, layout, read)
- Language specification
- Page selection
- Reading order control
- Caching support
- Support for pre-configured clients

## Notes

- Available models: "prebuilt-document", "prebuilt-layout", "prebuilt-read"
- Vision mode is supported for image formats
- Azure credentials are required
- Rate limits and quotas apply based on your Azure subscription

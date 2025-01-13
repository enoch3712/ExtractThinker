# Google Document AI Loader

The Google Document AI loader uses Google Cloud's Document AI service to extract text, tables, and structured information from documents.

## Supported Formats

- pdf
- jpeg/jpg
- png
- tiff
- gif
- bmp
- webp

## Usage

### Basic Usage

```python
from extract_thinker import DocumentLoaderGoogleDocumentAI

# Initialize with Google Cloud credentials
loader = DocumentLoaderGoogleDocumentAI(
    project_id="your_project_id",
    location="your_location",
    processor_id="your_processor_id",
    credentials_path="path/to/credentials.json"
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
from extract_thinker import DocumentLoaderGoogleDocumentAI, GoogleDocAIConfig

# Create configuration
config = GoogleDocAIConfig(
    project_id="your_project_id",
    location="your_location",
    processor_id="your_processor_id",
    credentials_path="path/to/credentials.json",
    mime_type="application/pdf",    # Specify MIME type
    process_options={               # Additional processing options
        "ocr_config": {"enable_native_pdf_parsing": True}
    },
    cache_ttl=600                   # Cache results for 10 minutes
)

# Initialize loader with configuration
loader = DocumentLoaderGoogleDocumentAI(config)

# Load and process document
pages = loader.load("path/to/your/document.pdf")
```

## Configuration Options

The `GoogleDocAIConfig` class supports the following options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `content` | Any | None | Initial content to process |
| `cache_ttl` | int | 300 | Cache time-to-live in seconds |
| `project_id` | str | None | Google Cloud project ID |
| `location` | str | None | Processing location |
| `processor_id` | str | None | Document AI processor ID |
| `credentials_path` | str | None | Path to credentials file |
| `credentials` | Credentials | None | Pre-configured credentials |
| `mime_type` | str | None | Document MIME type |
| `process_options` | Dict | None | Additional processing options |

## Features

- Text extraction with layout preservation
- Table detection and extraction
- Form field recognition
- Multiple processor support
- Native PDF parsing
- Custom processing options
- Caching support
- Support for pre-configured credentials

## Notes

- Vision mode is supported for image formats
- Google Cloud credentials are required
- Rate limits and quotas apply based on your Google Cloud account
- Different processors may support different document types
- Native PDF parsing can improve performance for PDF documents

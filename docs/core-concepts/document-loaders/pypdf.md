# PyPDF Document Loader

The PyPDF loader uses the PyPDF library to extract text and images from PDF documents. It provides basic text extraction and supports password-protected PDFs.

## Supported Formats

- pdf

## Usage

### Basic Usage

```python
from extract_thinker import DocumentLoaderPyPdf

# Initialize with default settings
loader = DocumentLoaderPyPdf()

# Load document
pages = loader.load("path/to/your/document.pdf")

# Process extracted content
for page in pages:
    # Access text content
    text = page["content"]
```

### Configuration-based Usage

```python
from extract_thinker import DocumentLoaderPyPdf, PyPDFConfig

# Create configuration
config = PyPDFConfig(
    password="your_password",      # For password-protected PDFs
    vision_enabled=True,           # Enable vision mode for images
    extract_text=True,             # Enable text extraction
    cache_ttl=600                  # Cache results for 10 minutes
)

# Initialize loader with configuration
loader = DocumentLoaderPyPdf(config)

# Load and process document
pages = loader.load("path/to/your/document.pdf")
```

## Configuration Options

The `PyPDFConfig` class supports the following options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `content` | Any | None | Initial content to process |
| `cache_ttl` | int | 300 | Cache time-to-live in seconds |
| `password` | str | None | Password for protected PDFs |
| `vision_enabled` | bool | False | Enable vision mode for images |
| `extract_text` | bool | True | Enable text extraction |

## Features

- Basic text extraction
- Password-protected PDF support
- Image extraction (with vision mode)
- Caching support
- No cloud service required
- Lightweight and fast processing

## Notes

- Vision mode can be enabled for image extraction
- Text extraction can be disabled for better performance
- Supports encrypted/password-protected PDFs
- Local processing with no external dependencies
- May not preserve complex layouts or tables
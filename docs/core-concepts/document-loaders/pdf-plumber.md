# PDFPlumber Document Loader

The PDFPlumber loader uses the pdfplumber library to extract text and tables from PDF documents with precise layout preservation.

## Supported Formats

- pdf

## Usage

### Basic Usage

```python
from extract_thinker import DocumentLoaderPdfPlumber

# Initialize with default settings
loader = DocumentLoaderPdfPlumber()

# Load document
pages = loader.load("path/to/your/document.pdf")

# Process extracted content
for page in pages:
    # Access text content
    text = page["content"]
    # Access tables if extracted
    tables = page.get("tables", [])
```

### Configuration-based Usage

```python
from extract_thinker import DocumentLoaderPdfPlumber, PDFPlumberConfig

# Create configuration
config = PDFPlumberConfig(
    table_settings={                # Custom table extraction settings
        "vertical_strategy": "text",
        "horizontal_strategy": "lines",
        "intersection_y_tolerance": 10
    },
    vision_enabled=True,           # Enable vision mode for images
    extract_tables=True,           # Enable table extraction
    cache_ttl=600                  # Cache results for 10 minutes
)

# Initialize loader with configuration
loader = DocumentLoaderPdfPlumber(config)

# Load and process document
pages = loader.load("path/to/your/document.pdf")
```

## Configuration Options

The `PDFPlumberConfig` class supports the following options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `content` | Any | None | Initial content to process |
| `cache_ttl` | int | 300 | Cache time-to-live in seconds |
| `table_settings` | Dict | None | Custom table extraction settings |
| `vision_enabled` | bool | False | Enable vision mode for images |
| `extract_tables` | bool | True | Enable table extraction |

## Features

- Text extraction with layout preservation
- Table detection and extraction
- Custom table extraction settings
- Vision mode support
- Precise positioning information
- Caching support
- No cloud service required

## Notes

- Vision mode can be enabled for image extraction
- Table extraction can be disabled for better performance
- Custom table settings can improve extraction accuracy
- Local processing with no external dependencies

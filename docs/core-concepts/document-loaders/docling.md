# Docling Document Loader

The Docling loader is a specialized document processor that excels at handling complex document layouts and table structures. It provides advanced OCR capabilities and precise table detection.

## Supported Formats

### Documents
- pdf
- doc/docx
- ppt/pptx
- xls/xlsx

### Images
- jpeg/jpg
- png
- tiff
- bmp
- gif
- webp

### Text
- txt
- html
- xml
- json

### Others
- csv
- tsv
- zip

## Usage

### Basic Usage

```python
from extract_thinker import DocumentLoaderDocling

# Initialize with default settings
loader = DocumentLoaderDocling()

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
from extract_thinker import DocumentLoaderDocling, DoclingConfig

# Create configuration
config = DoclingConfig(
    ocr_enabled=True,                # Enable OCR processing
    table_structure_enabled=True,    # Enable table structure detection
    tesseract_cmd="path/to/tesseract", # Custom Tesseract path
    force_full_page_ocr=False,      # Use selective OCR
    do_cell_matching=True,          # Enable cell content matching
    format_options={                # Format-specific options
        "pdf": {"dpi": 300},
        "image": {"enhance": True}
    },
    cache_ttl=600                   # Cache results for 10 minutes
)

# Initialize loader with configuration
loader = DocumentLoaderDocling(config)

# Load and process document
pages = loader.load("path/to/your/document.pdf")
```

## Configuration Options

The `DoclingConfig` class supports the following options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `content` | Any | None | Initial content to process |
| `cache_ttl` | int | 300 | Cache time-to-live in seconds |
| `ocr_enabled` | bool | False | Enable OCR processing |
| `table_structure_enabled` | bool | True | Enable table structure detection |
| `tesseract_cmd` | str | None | Path to Tesseract executable |
| `force_full_page_ocr` | bool | False | Force OCR on entire page |
| `do_cell_matching` | bool | True | Enable cell content matching |
| `format_options` | Dict | None | Format-specific processing options |

## Features

- Advanced table structure detection
- Selective OCR processing
- Cell content matching
- Format-specific optimizations
- Custom Tesseract integration
- Table content deduplication
- Multi-format support
- Caching support
- Stream-based loading

## Notes

- Vision mode is supported for image formats
- OCR requires Tesseract installation
- Table detection works best with structured documents
- Performance depends on document complexity
- Handles both scanned and digital documents
- Supports multiple document formats through format-specific optimizations 
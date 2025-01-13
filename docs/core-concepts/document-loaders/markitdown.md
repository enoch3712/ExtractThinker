# MarkItDown Document Loader

[MarkItDown](https://github.com/microsoft/markitdown) is a versatile document processing library from Microsoft that can handle multiple file formats. The MarkItDown loader provides a robust interface for text extraction with optional vision mode support.

## Supported Formats

### Documents
- pdf
- doc/docx
- ppt/pptx
- xls/xlsx

### Text
- txt
- html
- xml
- json

### Images
- jpg/jpeg
- png
- bmp
- gif

### Audio
- wav
- mp3
- m4a

### Others
- csv
- tsv
- zip

## Usage

### Basic Usage

```python
from extract_thinker import DocumentLoaderMarkItDown

# Initialize with default settings
loader = DocumentLoaderMarkItDown()

# Load document
pages = loader.load("path/to/your/document.pdf")

# Process extracted content
for page in pages:
    # Access text content
    text = page["content"]
```

### Configuration-based Usage

```python
from extract_thinker import DocumentLoaderMarkItDown, MarkItDownConfig

# Create configuration
config = MarkItDownConfig(
    page_separator="---",          # Custom page separator
    preserve_whitespace=True,      # Preserve original whitespace
    mime_type_detection=True,      # Enable MIME type detection
    default_extension=".md",       # Default file extension
    llm_client="gpt-4",           # LLM client for enhanced parsing
    cache_ttl=600                  # Cache results for 10 minutes
)

# Initialize loader with configuration
loader = DocumentLoaderMarkItDown(config)

# Load and process document
pages = loader.load("path/to/your/document.md")
```

## Configuration Options

The `MarkItDownConfig` class supports the following options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `content` | Any | None | Initial content to process |
| `cache_ttl` | int | 300 | Cache time-to-live in seconds |
| `page_separator` | str | "\n\n" | Text to use as page separator |
| `preserve_whitespace` | bool | False | Whether to preserve whitespace |
| `mime_type_detection` | bool | True | Enable MIME type detection |
| `default_extension` | str | ".txt" | Default file extension |
| `llm_client` | str | None | LLM client for enhanced parsing |
| `llm_model` | str | None | LLM model for enhanced parsing |

## Features

- Multi-format document processing
- Text and layout preservation
- MIME type detection
- Custom page separation
- Whitespace preservation
- LLM-enhanced parsing
- Caching support
- Stream-based loading

## Notes

- Vision mode is supported for image formats
- LLM enhancement is optional
- Local processing with no external dependencies
- Preserves document structure
- Handles a wide variety of file formats
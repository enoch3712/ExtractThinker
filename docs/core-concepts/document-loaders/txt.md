# Text File Document Loader

The Text File loader is a simple loader for reading plain text files. It has no external dependencies as it uses Python's built-in file handling.

## Supported Formats

- txt

## Usage

### Basic Usage

```python
from extract_thinker import DocumentLoaderTxt

# Initialize the loader with default settings
loader = DocumentLoaderTxt()

# Load document
pages = loader.load("path/to/your/document.txt")

# Process extracted content
for page in pages:
    # Access text content
    text = page["content"]
```

### Configuration-based Usage

```python
from extract_thinker import DocumentLoaderTxt, TxtConfig

# Create configuration
config = TxtConfig(
    encoding='utf-8',              # Specify text encoding
    preserve_whitespace=True,      # Preserve original whitespace
    split_paragraphs=True,         # Split text into paragraphs
    cache_ttl=600                  # Cache results for 10 minutes
)

# Initialize loader with configuration
loader = DocumentLoaderTxt(config)

# Load and process document
pages = loader.load("path/to/your/document.txt")
```

## Configuration Options

The `TxtConfig` class supports the following options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `content` | Any | None | Initial content to process |
| `cache_ttl` | int | 300 | Cache time-to-live in seconds |
| `encoding` | str | 'utf-8' | Text encoding to use |
| `preserve_whitespace` | bool | False | Whether to preserve whitespace in text |
| `split_paragraphs` | bool | False | Whether to split text into paragraphs |

## Features

- Simple text file reading
- Configurable text encoding
- Whitespace preservation control
- Paragraph splitting option
- Stream-based loading support
- Caching support
- No external dependencies required

## Notes

- Vision mode is not supported for text files
- BytesIO streams are supported for in-memory processing
- Default encoding is UTF-8
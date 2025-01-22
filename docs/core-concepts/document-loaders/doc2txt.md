# Doc2txt Document Loader

The Doc2txt loader extracts text from Microsoft Word documents. It supports both legacy (.doc) and modern (.docx) file formats.

## Supported Formats

- doc
- docx

## Usage

### Basic Usage

```python
from extract_thinker import DocumentLoaderDoc2txt

# Initialize with default settings
loader = DocumentLoaderDoc2txt()

# Load document
pages = loader.load("path/to/your/document.docx")

# Process extracted content
for page in pages:
    # Access text content
    text = page["content"]
```

### Configuration-based Usage

```python
from extract_thinker import DocumentLoaderDoc2txt, Doc2txtConfig

# Create configuration
config = Doc2txtConfig(
    page_separator="\n\n---\n\n",  # Custom page separator
    preserve_whitespace=True,      # Preserve original whitespace
    extract_images=True,           # Extract embedded images
    cache_ttl=600                  # Cache results for 10 minutes
)

# Initialize loader with configuration
loader = DocumentLoaderDoc2txt(config)

# Load and process document
pages = loader.load("path/to/your/document.docx")
```

## Configuration Options

The `Doc2txtConfig` class supports the following options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `content` | Any | None | Initial content to process |
| `cache_ttl` | int | 300 | Cache time-to-live in seconds |
| `page_separator` | str | "\n\n" | Text to use as page separator |
| `preserve_whitespace` | bool | False | Whether to preserve whitespace |
| `extract_images` | bool | False | Whether to extract embedded images |

## Features

- Text extraction from Word documents
- Support for both .doc and .docx
- Custom page separation
- Whitespace preservation
- Image extraction (optional)
- Caching support
- No cloud service required

## Notes

- Vision mode is not supported
- Image extraction requires additional memory
- Local processing with no external dependencies
- May not preserve complex formatting
- Handles both legacy and modern Word formats
# Text File Loader

The Text File loader is a simple but effective loader designed to handle plain text files (`.txt`). It provides basic text extraction capabilities while maintaining compatibility with the ExtractThinker framework.

## Basic Usage

```python
from extract_thinker import Extractor, DocumentLoaderTxt

# Initialize the extractor with Text loader
extractor = Extractor()
extractor.load_document_loader(DocumentLoaderTxt())

# Process a text file
result = extractor.extract("document.txt", YourContract)
```

## Features

- Handles plain text files (`.txt`)
- UTF-8 encoding support
- Treats the entire file as a single content block
- Caches results for improved performance
- Supports both file paths and BytesIO streams

## Limitations

- Does not support vision mode
- Limited to plain text files only
- No formatting preservation (since plain text files don't have formatting)

## Configuration

The loader can be configured with caching options:

```python
# Configure with custom cache TTL (in seconds)
loader = DocumentLoaderTxt(cache_ttl=600)  # 10 minutes cache
``` 
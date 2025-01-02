# Text File Document Loader

The Text File loader is a simple loader for reading plain text files. It has no external dependencies as it uses Python's built-in file handling.

## Supported Formats

- txt

## Usage

```python
from extract_thinker import DocumentLoaderTxt

# Initialize the loader
loader = DocumentLoaderTxt()

# Load document
pages = loader.load("path/to/your/document.txt")

# Process extracted content
for page in pages:
    # Access text content
    text = page["content"]
```

## Features

- Simple text file reading
- UTF-8 encoding support
- Stream-based loading support
- No external dependencies required
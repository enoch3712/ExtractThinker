# MarkItDown Document Loader

[MarkItDown](https://github.com/microsoft/markitdown) is a versatile document processing library from Microsoft that can handle multiple file formats. ExtractThinker's MarkItDown loader provides a robust interface for text extraction with optional vision mode support.

## Basic Usage

Here's how to use the MarkItDown loader:

```python
from extract_thinker import Extractor
from extract_thinker.document_loader import DocumentLoaderMarkItDown

# Initialize the loader
loader = DocumentLoaderMarkItDown()

# Load document content
pages = loader.load("document.pdf")

# Access text content from first page
text = pages[0]["content"]

# Enable vision mode for image extraction
loader.set_vision_mode(True)
pages_with_images = loader.load("document.pdf")

# Access both text and image
text = pages_with_images[0]["content"]
image = pages_with_images[0]["image"]  # bytes object
```

## Features

- Multi-format support (`PDF`, `DOC`, `DOCX`, `PPT`, `PPTX`, `XLS`, `XLSX`, etc.)
- Text extraction from various file types
- Optional vision mode for image extraction
- Page-by-page processing
- Stream-based loading support
- Caching capabilities
- LLM integration support

## Supported Formats

- Documents: `PDF`, `DOC`, `DOCX`, `PPT`, `PPTX`, `XLS`, `XLSX`
- Text: `TXT`, `HTML`, `XML`, `JSON`
- Images: `JPG`, `JPEG`, `PNG`, `BMP`, `GIF`
- Audio: `WAV`, `MP3`, `M4A`
- Others: `CSV`, `TSV`, `ZIP`
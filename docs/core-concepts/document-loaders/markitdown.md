# MarkItDown Document Loader

MarkItDown is a versatile document processing library that can handle multiple file formats. ExtractThinker's MarkItDown loader provides a robust interface for text extraction with optional vision mode support.

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

- Multi-format support (PDF, DOC, DOCX, PPT, PPTX, XLS, XLSX, etc.)
- Text extraction from various file types
- Optional vision mode for image extraction
- Page-by-page processing
- Stream-based loading support
- Caching capabilities
- LLM integration support

## Supported Formats

- Documents: PDF, DOC, DOCX, PPT, PPTX, XLS, XLSX
- Text: TXT, HTML, XML, JSON
- Images: JPG, JPEG, PNG, BMP, GIF
- Audio: WAV, MP3, M4A
- Others: CSV, TSV, ZIP

## Best Practices

1. **Document Processing**
   - Use vision mode only when image extraction is needed
   - Enable caching for repeated processing
   - Handle large documents using stream-based loading

2. **Performance**
   - Configure cache TTL based on your needs
   - Monitor memory usage with large files
   - Use appropriate file formats for best results

3. **LLM Integration**
   - Provide LLM client and model when needed
   - Configure based on your specific use case
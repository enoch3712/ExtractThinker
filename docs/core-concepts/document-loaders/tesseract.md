# Tesseract Document Loader

> Tesseract is an open-source OCR engine that can extract text from images and scanned PDFs. ExtractThinker's Tesseract Document Loader provides a simple interface to use Tesseract for document processing.

## Prerequisite

You need to have Tesseract installed on your system and set the path to the Tesseract executable:

```bash
# On Ubuntu/Debian
sudo apt-get install tesseract-ocr

# On macOS
brew install tesseract

# On Windows
# Download installer from https://github.com/UB-Mannheim/tesseract/wiki
```

```python
%pip install --upgrade --quiet extract_thinker pytesseract
```

## Basic Usage

Here's a simple example of using the Tesseract Document Loader:

```python
from extract_thinker import Extractor
from extract_thinker.document_loader import DocumentLoaderTesseract

# Initialize the loader with Tesseract path
tesseract_path = os.getenv("TESSERACT_PATH")
loader = DocumentLoaderTesseract(tesseract_path)

# Load document
content = loader.load("invoice.png")

# Get content list (page by page)
content_list = loader.load_content_list("invoice.png")
```

Supports `PDF`, `JPEG/JPG`, `PNG`, `BMP`, `TIFF`

## Best Practices

- Ensure good image quality for optimal results
- Use appropriate language packs for non-English documents
- Consider image preprocessing for better accuracy
- Set appropriate page segmentation mode based on document layout

For more examples and advanced usage, check out the [Local Stack](../../../examples/local-processing) in the repository.

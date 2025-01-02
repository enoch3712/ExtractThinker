# Tesseract Document Loader

> Tesseract is an open-source OCR engine that can extract text from images and scanned PDFs. ExtractThinker's Tesseract Document Loader provides a simple interface to use Tesseract for document processing.

## Installation

1. First, install Tesseract OCR on your system:

   **Ubuntu/Debian:**
   ```bash
   sudo apt-get install tesseract-ocr
   ```

   **macOS:**
   ```bash
   brew install tesseract
   ```

   **Windows:**
   Download and install from [GitHub Tesseract releases](https://github.com/UB-Mannheim/tesseract/wiki)

2. Install the Python package:
   ```bash
   pip install pytesseract Pillow
   ```

## Basic Usage

Here's a simple example of using the Tesseract Document Loader:

```python
from extract_thinker import DocumentLoaderTesseract

# Initialize the loader with Tesseract path
tesseract_path = os.getenv("TESSERACT_PATH")
loader = DocumentLoaderTesseract(tesseract_path)

# Load from file
pages = loader.load("path/to/your/image.png")

# Process the extracted text
for page in pages:
    text = page["content"]
    print(f"Extracted text: {text}")
```

Supports `PDF`, `JPEG/JPG`, `PNG`, `BMP`, `TIFF`

## Best Practices

- Ensure good image quality for optimal results
- Use appropriate language packs for non-English documents
- Consider image preprocessing for better accuracy
- Set appropriate page segmentation mode based on document layout

For more examples and advanced usage, check out the [Local Stack](../../../examples/local-processing) in the repository.

# LLM Image Loader

The LLM Image loader is a specialized loader designed to handle images and PDFs for vision-enabled Language Models. It serves as a fallback loader when no other loader is available and vision mode is required.

## Basic Usage

```python
from extract_thinker import Extractor, DocumentLoaderLLMImage

# Initialize the extractor with LLM Image loader
extractor = Extractor()
extractor.load_document_loader(DocumentLoaderLLMImage())

# Process an image or PDF
result = extractor.extract("document.png", YourContract)
```

## Features

- Supports multiple image formats: `.pdf`, `.jpg`, `.jpeg`, `.png`, `.tiff`, `.bmp`
- Always operates in vision mode
- Preserves image data for LLM processing
- Caches results for improved performance
- Can handle both file paths and BytesIO streams

## Use Cases

- Processing documents where text extraction is difficult or unreliable
- Working with image-heavy documents
- Using vision-enabled LLMs for document understanding
- Fallback option when other loaders fail

## Notes

- This loader is specifically designed for vision/image processing
- It doesn't extract text content (content field will be empty)
- Each page will contain the image data in the 'image' field
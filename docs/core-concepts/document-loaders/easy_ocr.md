# EasyOCR Document Loader

The EasyOCR loader uses the EasyOCR engine to extract text from images and PDF files. It's known for its ease of use and support for a wide range of languages.

## Supported Formats

- png
- jpg/jpeg
- tiff/tif
- webp
- pdf

## Usage

### Basic Usage

```python
from extract_thinker import DocumentLoaderEasyOCR, EasyOCRConfig

# Initialize with default settings
config = EasyOCRConfig()
loader = DocumentLoaderEasyOCR(config)

# Load document
pages = loader.load("path/to/your/image.png")

# Process extracted content
for page in pages:
    # Access text content
    text = page["content"]
    print(text)
```

### Configuration-based Usage

```python
from extract_thinker import DocumentLoaderEasyOCR, EasyOCRConfig

# Create configuration
config = EasyOCRConfig(
    lang_list=['en', 'fr'],        # Use English and French
    gpu=True,                      # Enable GPU acceleration
    cache_ttl=600,                 # Cache results for 10 minutes
    include_bbox=True              # Include bounding box details
)

# Initialize loader with configuration
loader = DocumentLoaderEasyOCR(config)

# Load a PDF document
pages = loader.load("path/to/your/document.pdf")

# Process extracted content with details
for page in pages:
    print(f"Content: {page['content']}")
    if 'detail' in page:
        for detail in page['detail']:
            print(f"  - Text: {detail['text']}, BBox: {detail['bbox']}")

```

## Configuration Options

The `EasyOCRConfig` class supports the following options:

| Option | Type | Default | Description |
|--------------------|--------------|--------------------|-----------------------------------------------------------------|
| `lang_list` | List[str] | `['en']` | List of language codes for OCR (e.g., `['en', 'es']`). |
| `gpu` | bool | `True` | Whether to use GPU for processing (if available). |
| `download_enabled` | bool | `True` | Automatically download language models if not found. |
| `cache_ttl` | int | `300` | Time-to-live for cached results, in seconds. |
| `include_bbox` | bool | `False` | Whether to include detailed bounding box information in the output. |

## Features

- Text extraction from images and PDFs
- Multi-language support with automatic model downloading
- Optional inclusion of detailed bounding box data
- GPU acceleration for faster processing
- Caching support to improve performance for repeated requests
- Local processing with no external API calls

## Installation

EasyOCR requires the `easyocr` and `torch` libraries. You can install them using pip:

```bash
pip install easyocr torch
```
For GPU support, you may need to install a specific version of PyTorch that matches your CUDA version. Please refer to the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) for more details.

## Notes

- Vision mode is not supported by this loader.
- Performance is significantly better when a GPU is available.
- The first time a language is used, the corresponding model will be downloaded, which may take some time. 
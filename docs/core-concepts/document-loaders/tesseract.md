# Tesseract Document Loader

The Tesseract loader uses the Tesseract OCR engine to extract text from images. It supports multiple languages and provides various OCR optimization options.

## Supported Formats

- jpeg/jpg
- png
- tiff
- bmp
- gif

## Usage

### Basic Usage

```python
from extract_thinker import DocumentLoaderTesseract

# Initialize with default settings
loader = DocumentLoaderTesseract()

# Load document
pages = loader.load("path/to/your/image.png")

# Process extracted content
for page in pages:
    # Access text content
    text = page["content"]
```

### Configuration-based Usage

```python
from extract_thinker import DocumentLoaderTesseract, TesseractConfig

# Create configuration
config = TesseractConfig(
    lang="eng+fra",                # Use English and French
    psm=6,                         # Assume uniform block of text
    oem=3,                         # Default LSTM OCR Engine Mode
    config_params={                # Additional Tesseract parameters
        "tessedit_char_whitelist": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    },
    timeout=30,                    # OCR timeout in seconds
    cache_ttl=600                  # Cache results for 10 minutes
)

# Initialize loader with configuration
loader = DocumentLoaderTesseract(config)

# Load and process document
pages = loader.load("path/to/your/image.png")
```

## Configuration Options

The `TesseractConfig` class supports the following options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `content` | Any | None | Initial content to process |
| `cache_ttl` | int | 300 | Cache time-to-live in seconds |
| `lang` | str | "eng" | Language(s) for OCR |
| `psm` | int | 3 | Page segmentation mode |
| `oem` | int | 3 | OCR Engine Mode |
| `config_params` | Dict | None | Additional Tesseract parameters |
| `timeout` | int | 0 | OCR timeout in seconds |

## Features

- Text extraction from images
- Multi-language support
- Configurable page segmentation
- Multiple OCR engine modes
- Custom Tesseract parameters
- Timeout control
- Caching support
- No cloud service required

## Notes

- Vision mode is always enabled
- Requires Tesseract installation
- Performance depends on image quality
- Local processing with no external API calls
- Language data files must be installed separately

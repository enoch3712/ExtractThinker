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

## 🪟 Windows Installation

If you're using Windows, follow these steps to install Tesseract OCR:

1. Download the Tesseract installer from [UB Mannheim's GitHub repository](https://github.com/UB-Mannheim/tesseract/wiki)
2. Choose the appropriate installer:
   - For 64-bit Windows: `tesseract-ocr-w64-setup-xxx.exe`
   - For 32-bit Windows: `tesseract-ocr-w32-setup-xxx.exe`
3. During installation:
   - Choose the default installation path (`C:\Program Files\Tesseract-OCR`)
   - **Important**: Check the box for "Add to system PATH"
   - Complete the installation
4. Set up environment variables by creating a `.env` file in your project's root directory:
```
TESSERACT_PATH="C:\Program Files\Tesseract-OCR\tesseract.exe"
```
5. Verify installation by opening a new PowerShell window and running:
```powershell
where.exe tesseract
```

## Notes

- Vision mode is always enabled
- Requires Tesseract installation
- Performance depends on image quality
- Local processing with no external API calls
- Language data files must be installed separately

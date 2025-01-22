# LLM Image Document Loader

The LLM Image loader is a specialized loader designed to handle images and PDFs for vision-enabled Language Models. It serves as a fallback loader when no other loader is available and vision mode is required.

## Supported Formats

- jpeg/jpg
- png
- gif
- bmp
- webp
- tiff

## Usage

### Basic Usage

```python
from extract_thinker import DocumentLoaderLLMImage

# Initialize with default settings
loader = DocumentLoaderLLMImage()

# Load document
pages = loader.load("path/to/your/image.jpg")

# Process extracted content
for page in pages:
    # Access image content
    image_bytes = page["image"]
    # Access metadata if available
    metadata = page.get("metadata", {})
```

### Configuration-based Usage

```python
from extract_thinker import DocumentLoaderLLMImage, LLMImageConfig

# Create configuration
config = LLMImageConfig(
    max_image_size=1024 * 1024,    # Maximum image size in bytes
    image_format="jpeg",           # Target image format
    compression_quality=85,        # JPEG compression quality
    llm="gpt-4-vision",           # Target LLM model
    cache_ttl=600                  # Cache results for 10 minutes
)

# Initialize loader with configuration
loader = DocumentLoaderLLMImage(config)

# Load and process document
pages = loader.load("path/to/your/image.jpg")
```

## Configuration Options

The `LLMImageConfig` class supports the following options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `content` | Any | None | Initial content to process |
| `cache_ttl` | int | 300 | Cache time-to-live in seconds |
| `llm` | str | None | Target LLM model |
| `max_image_size` | int | 1048576 | Maximum image size in bytes |
| `image_format` | str | "jpeg" | Target image format |
| `compression_quality` | int | 85 | JPEG compression quality |

## Features

- Processing documents where text extraction is difficult or unreliable
- Working with image-heavy documents
- Using vision-enabled LLMs for document understanding
- Fallback option when other loaders fail

## Notes

- This loader is specifically designed for vision/image processing
- It doesn't extract text content (content field will be empty)
- Each page will contain the image data in the 'image' field
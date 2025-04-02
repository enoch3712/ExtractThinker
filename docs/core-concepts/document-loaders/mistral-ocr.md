# Mistral OCR Document Loader

The Mistral OCR document loader leverages the Mistral OCR API to extract text and images from various document formats. It provides high-quality OCR capabilities with advanced machine learning models.

## About Mistral OCR

Mistral OCR is an industry-leading Optical Character Recognition API that sets a new standard in document understanding. Unlike other models, Mistral OCR comprehends each element of documents—media, text, tables, equations—with unprecedented accuracy and cognition. It takes images and PDFs as input and extracts content in an ordered interleaved text and images format.

Key capabilities:
- State-of-the-art understanding of complex documents (tables, equations, layouts)
- Natively multilingual support across thousands of scripts and languages
- Superior performance on benchmarks
- Fast processing (up to 2000 pages per minute on a single node)
- Structured output in markdown format

## Performance Benchmarks

Mistral OCR consistently outperforms other leading OCR models in benchmark tests:

| Model                | Overall | Math  | Multilingual | Scanned | Tables |
| -------------------- | ------- | ----- | ------------ | ------- | ------ |
| Google Document AI   | 83.42   | 80.29 | 86.42        | 92.77   | 78.16  |
| Azure OCR            | 89.52   | 85.72 | 87.52        | 94.65   | 89.52  |
| Gemini-1.5-Flash-002 | 90.23   | 89.11 | 86.76        | 94.87   | 90.48  |
| Gemini-1.5-Pro-002   | 89.92   | 88.48 | 86.33        | 96.15   | 89.71  |
| Gemini-2.0-Flash-001 | 88.69   | 84.18 | 85.80        | 95.11   | 91.46  |
| GPT-4o-2024-11-20    | 89.77   | 87.55 | 86.00        | 94.58   | 91.70  |
| Mistral OCR 2503     | 94.89   | 94.29 | 89.55        | 98.96   | 96.12  |

### Multilingual Performance

Mistral OCR excels at processing documents in multiple languages:

| Language | Azure OCR | Google Doc AI | Gemini-2.0-Flash-001 | Mistral OCR 2503 |
| -------- | --------- | ------------- | -------------------- | ---------------- |
| ru       | 97.35     | 95.56         | 96.58                | 99.09            |
| fr       | 97.50     | 96.36         | 97.06                | 99.20            |
| hi       | 96.45     | 95.65         | 94.99                | 97.55            |
| zh       | 91.40     | 90.89         | 91.85                | 97.11            |
| pt       | 97.96     | 96.24         | 97.25                | 99.42            |
| de       | 98.39     | 97.09         | 97.19                | 99.51            |
| es       | 98.54     | 97.52         | 97.75                | 99.54            |
| tr       | 95.91     | 93.85         | 94.66                | 97.00            |
| uk       | 97.81     | 96.24         | 96.70                | 99.29            |
| it       | 98.31     | 97.69         | 97.68                | 99.42            |
| ro       | 96.45     | 95.14         | 95.88                | 98.79            |

## Supported Formats

- PDF documents
- Image files:
  - JPG/JPEG
  - PNG
  - TIFF
  - BMP

## Usage

### Basic Usage

```python
from extract_thinker import DocumentLoaderMistralOCR, MistralOCRConfig

# Create configuration
config = MistralOCRConfig(
    api_key="your_mistral_api_key",
    model="mistral-ocr-latest"
)

# Initialize loader
loader = DocumentLoaderMistralOCR(config)

# Load from URL
pages = loader.load("https://example.com/document.pdf")

# Load from file path
pages = loader.load("path/to/your/document.pdf")

# Process extracted content
for page in pages:
    # Access text content (in markdown format)
    markdown_text = page["content"]
    
    # Access images if available
    if "images" in page:
        for image in page["images"]:
            image_id = image["id"]
            image_base64 = image["image_base64"]  # If include_image_base64=True
```

### Configuration Options

The `MistralOCRConfig` class supports the following options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `api_key` | str | Required | Mistral API key |
| `model` | str | "mistral-ocr-latest" | OCR model to use |
| `content` | Any | None | Initial content to process |
| `cache_ttl` | int | 300 | Cache time-to-live in seconds |
| `include_image_base64` | bool | False | Include image base64 in response |
| `pages` | List[int] | None | Specific pages to process (PDF only) |
| `image_limit` | int | None | Maximum number of images to extract |
| `image_min_size` | int | None | Minimum image size to extract |

## Features

- High-quality OCR with Mistral AI's models
- Support for PDF and image formats
- Text extraction in markdown format
- Image extraction with positioning information
- Support for pagination in PDF documents
- Caching for improved performance
- URL, file path, and BytesIO input support
- Processing speed up to 2000 pages per minute
- Superior handling of complex elements like tables, math equations, and diagrams
- Native support for thousands of languages and scripts

## How It Works

When processing a document with the Mistral OCR loader:

1. For URLs: The URL is sent directly to the Mistral OCR API
2. For file paths or BytesIO objects:
   - The file is first uploaded to Mistral's file storage system
   - A signed URL is generated for the uploaded file
   - The OCR API processes the document using the signed URL

This approach follows Mistral's recommended workflow for document processing and complies with their API requirements.

## Common Use Cases

Mistral OCR can be used for a variety of document processing tasks:

- **Scientific research**: Convert scientific papers with complex equations and diagrams into AI-ready formats
- **Historical document preservation**: Digitize historical documents and artifacts
- **Customer service enhancement**: Transform documentation and manuals into indexed knowledge
- **Educational content processing**: Extract information from lecture notes, presentations, and educational materials
- **Legal document analysis**: Process regulatory filings and legal documents with high accuracy
- **Multilingual document handling**: Process documents in multiple languages with superior accuracy

## API Usage Notes

- The Mistral OCR API requires authentication with an API key
- API usage is subject to Mistral AI's terms and pricing (approximately 1000 pages / $)
- Response time depends on document size and complexity
- Extracted text is returned in markdown format
- Image positions and dimensions are provided for visual context
- Pagination is only supported for PDF documents
- Maximum document size: 50 MB
- Maximum page limit: 1,000 pages
- Local files are uploaded to Mistral's file storage with a purpose of "ocr"

## Requirements

- An active Mistral AI API key
- `requests` library for API communication
- Internet connectivity for API access 
# AWS Textract Document Loader

The AWS Textract loader uses Amazon's Textract service to extract text, forms, and tables from documents. It supports both image files and PDFs.

## Supported Formats

- pdf
- jpeg
- png
- tiff

## Usage

### Basic Usage

```python
from extract_thinker import DocumentLoaderAWSTextract

# Initialize with AWS credentials
loader = DocumentLoaderAWSTextract(
    aws_access_key_id="your_access_key",
    aws_secret_access_key="your_secret_key",
    region_name="your_region"
)

# Load document
pages = loader.load("path/to/your/document.pdf")

# Process extracted content
for page in pages:
    # Access text content
    text = page["content"]
    # Access tables if extracted
    tables = page.get("tables", [])
```

### Configuration-based Usage

```python
from extract_thinker import DocumentLoaderAWSTextract, TextractConfig

# Create configuration
config = TextractConfig(
    aws_access_key_id="your_access_key",
    aws_secret_access_key="your_secret_key",
    region_name="your_region",
    feature_types=["TABLES", "FORMS", "SIGNATURES"],  # Specify features to extract
    cache_ttl=600,                                    # Cache results for 10 minutes
    max_retries=3                                     # Number of retry attempts
)

# Initialize loader with configuration
loader = DocumentLoaderAWSTextract(config)

# Load and process document
pages = loader.load("path/to/your/document.pdf")
```

## Configuration Options

The `TextractConfig` class supports the following options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `content` | Any | None | Initial content to process |
| `cache_ttl` | int | 300 | Cache time-to-live in seconds |
| `aws_access_key_id` | str | None | AWS access key ID |
| `aws_secret_access_key` | str | None | AWS secret access key |
| `region_name` | str | None | AWS region name |
| `textract_client` | boto3.client | None | Pre-configured Textract client |
| `feature_types` | List[str] | [] | Features to extract (TABLES, FORMS, LAYOUT, SIGNATURES) |
| `max_retries` | int | 3 | Maximum number of retry attempts |

## Features

- Text extraction from images and PDFs
- Table detection and extraction
- Form field detection
- Layout analysis
- Signature detection
- Configurable feature selection
- Automatic retry on failure
- Caching support
- Support for pre-configured clients

## Notes

- Raw text extraction is the default when no feature types are specified
- "QUERIES" feature type is not supported
- Vision mode is supported for image formats
- AWS credentials are required unless using a pre-configured client
- Rate limits and quotas apply based on your AWS account 
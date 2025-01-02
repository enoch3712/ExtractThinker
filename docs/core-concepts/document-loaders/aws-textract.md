# AWS Textract Document Loader

> AWS Textract provides advanced OCR and document analysis capabilities, extracting text, forms, and tables from documents.

## Installation

Install the required dependencies:

```bash
pip install boto3
```

## Prerequisites

1. An AWS account
2. AWS credentials with access to Textract service
3. AWS region where Textract is available

## Supported Formats

- Images: jpeg/jpg, png, tiff
- Documents: pdf

## Usage

```python
from extract_thinker import DocumentLoaderAWSTextract

# Initialize the loader with AWS credentials
loader = DocumentLoaderAWSTextract(
    aws_access_key_id="your-access-key",
    aws_secret_access_key="your-secret-key",
    region_name="your-region"
)

# Load document content
result = loader.load_content_from_file("document.pdf")
```

## Response Structure

The loader returns a dictionary with the following structure:

```python
{
    "pages": [
        {
            "paragraphs": ["text content..."],
            "lines": ["line1", "line2"],
            "words": ["word1", "word2"]
        }
    ],
    "tables": [
        [["cell1", "cell2"], ["cell3", "cell4"]]
    ],
    "forms": [
        {"key": "value"}
    ],
    "layout": {
        # Document layout information
    }
}
```

## Supported Formats

`PDF`, `JPEG`, `PNG`

## Features

- Text extraction with layout preservation
- Table detection and extraction
- Support for multiple document formats
- Automatic retries on API failures 
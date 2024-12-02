# AWS Textract Document Loader

> AWS Textract provides advanced OCR and document analysis capabilities, extracting text, forms, and tables from documents.

## Prerequisite

You need AWS credentials with access to Textract service. You will need:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`

```python
%pip install --upgrade --quiet extract_thinker boto3
```

## Basic Usage

Here's a simple example of using the AWS Textract loader:

```python
from extract_thinker import DocumentLoaderTextract

# Initialize the loader
loader = DocumentLoaderTextract(
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_DEFAULT_REGION')
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

## Best Practices

1. **Document Preparation**
   - Use high-quality scans
   - Support formats: PDF, JPEG, PNG
   - Consider file size limits

2. **Performance**
   - Cache results when possible
   - Process pages individually for large documents
   - Monitor API quotas and costs

For more examples and implementation details, check out the [AWS Stack](../../examples/aws-textract) in the repository. 
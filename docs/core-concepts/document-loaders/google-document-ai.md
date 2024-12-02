# Google Document AI Document Loader

> Google Document AI transforms unstructured document data into structured, actionable insights using machine learning.

## Prerequisite

You need Google Cloud credentials and a Document AI processor. You will need:
- `DOCUMENTAI_GOOGLE_CREDENTIALS`
- `DOCUMENTAI_LOCATION`
- `DOCUMENTAI_PROCESSOR_NAME`

```python
%pip install --upgrade --quiet extract_thinker google-cloud-documentai
```

## Basic Usage

Here's a simple example of using the Google Document AI loader:

```python
from extract_thinker import DocumentLoaderDocumentAI

# Initialize the loader
loader = DocumentLoaderDocumentAI(
    credentials=os.getenv("DOCUMENTAI_GOOGLE_CREDENTIALS"),
    location=os.getenv("DOCUMENTAI_LOCATION"),
    processor_name=os.getenv("DOCUMENTAI_PROCESSOR_NAME")
)

# Load CV/Resume content
content = loader.load_content_from_file("CV_Candidate.pdf")
```

## Response Structure

The loader returns a dictionary containing:
```python
{
    "pages": [
        {
            "content": "Full text content of the page",
            "paragraphs": ["Paragraph 1", "Paragraph 2"],
            "tables": [
                [
                    ["Header 1", "Header 2"],
                    ["Value 1", "Value 2"]
                ]
            ]
        }
    ]
}
```

## Processing Different Document Types

```python
# Process forms with tables
content = loader.load_content_from_file("form_with_tables.pdf")

# Process from stream
with open("document.pdf", "rb") as f:
    content = loader.load_content_from_stream(
        stream=f,
        mime_type="application/pdf"
    )
```

## Best Practices

1. **Document Types**
   - Use appropriate processor for document type
   - Ensure correct MIME type for streams
   - Validate content structure

2. **Performance**
   - Process in batches when possible
   - Cache results for repeated access
   - Monitor API quotas

Document AI supports PDF, TIFF, GIF, JPEG, PNG with a maximum file size of 20MB or 2000 pages.

For more examples and implementation details, check out the [Google Stack](../../examples/google-document-ai) in the repository. 
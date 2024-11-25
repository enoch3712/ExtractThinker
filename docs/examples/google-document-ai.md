# Google Document AI Example

This guide shows how to use Google Document AI for advanced document processing.

## Basic Setup

Here's how to use Google Document AI:

```python
from extract_thinker import Extractor, Contract, DocumentLoaderDocumentAI
from typing import List
from pydantic import Field

class InvoiceContract(Contract):
    invoice_number: str = Field("Invoice number")
    invoice_date: str = Field("Invoice date")
    total_amount: float = Field("Total amount")
    lines: List[LineItem] = Field("List of line items")

# Initialize Google Document AI
extractor = Extractor()
extractor.load_document_loader(
    DocumentLoaderDocumentAI(
        credentials=os.getenv("DOCUMENTAI_GOOGLE_CREDENTIALS"),
        location=os.getenv("DOCUMENTAI_LOCATION"),
        processor_name=os.getenv("DOCUMENTAI_PROCESSOR_NAME")
    )
)

# Configure model
extractor.load_llm("gpt-4o")

# Process document
result = extractor.extract("invoice.pdf", InvoiceContract)
```

## Advanced Processing

Document AI offers specialized processors:

```python
# Process forms
result = loader.process_document(
    document="form.pdf",
    processor_type="FORM_PARSER_PROCESSOR"
)

# Process invoices
result = loader.process_document(
    document="invoice.pdf",
    processor_type="INVOICE_PROCESSOR"
)

# Process custom documents
result = loader.process_document(
    document="custom.pdf",
    processor_type="CUSTOM_PROCESSOR"
)
```

## Cost Optimization

Document AI pricing:
- OCR: $0.05 per page
- Specialized processors: $0.05-$0.10 per page
- Custom processors: Higher pricing

## Best Practices

1. **Processor Selection**
   - Use specialized processors when available
   - Train custom processors for specific needs
   - Consider OCR for basic text extraction

2. **Data Preparation**
   - Ensure high-quality document scans
   - Use supported file formats
   - Consider document preprocessing

3. **Error Handling**
   ```python
   try:
       result = extractor.extract(
           "document.pdf",
           Contract,
           processor_options={
               "enable_native_pdf_parsing": True
           }
       )
   except Exception as e:
       print(f"Processing error: {e}")
   ```

## Performance Metrics

- Processing speed: 1-2 seconds per page
- Accuracy: >98% for supported document types
- Supported formats: PDF, TIFF, GIF, JPEG, PNG
- Maximum file size: 20MB or 2000 pages
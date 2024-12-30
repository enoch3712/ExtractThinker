# Processing Documents with Groq

> ⚠️ **Warning**: Vision-based processing may not be available with Groq models. For image or document processing that requires vision capabilities, consider using other providers like Google Document AI, Azure Document Intelligence, or AWS Textract.

This guide demonstrates how to process documents using Groq's powerful LLMs.

## Basic Setup

Here's a basic example of document extraction using Groq:

```python
from extract_thinker import Extractor
from extract_thinker.document_loader.document_loader_pypdf import DocumentLoaderPyPdf
from typing import List
from pydantic import Field

class InvoiceContract(Contract):
    lines: List[LineItem] = Field("List of line items in the invoice")

# Initialize extractor with PyPDF loader
extractor = Extractor()
extractor.load_document_loader(DocumentLoaderPyPdf())

# Configure Groq
extractor.load_llm("groq/llama-3.2-11b-vision-preview")

# Process document
result = extractor.extract("invoice.pdf", InvoiceContract)
```

## Classification Example

You can also use Groq for document classification:

```python
from extract_thinker import Process
from extract_thinker.models.classification import Classification

# Setup process
process = Process()
process.add_classify_extractor([[extractor]])

# Define classifications
classifications = [
    Classification(
        name="Invoice",
        description="This is an invoice document", 
        contract=InvoiceContract
    ),
    Classification(
        name="Driver License",
        description="This is a driver license document",
        contract=DriverLicense
    )
]

# Classify document
result = process.classify("document.pdf", classifications)
```

**Benefits - High Performance**

- Fast inference times
- State-of-the-art language models
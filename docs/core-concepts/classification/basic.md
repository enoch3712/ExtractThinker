# Basic Classification

When classifying documents, the process involves extracting the content of the document and adding it to the prompt with several possible classifications. ExtractThinker simplifies this process using Pydantic models and instructor.

## Simple Classification

The most straightforward way to classify documents:

```python
from extract_thinker import Classification, Extractor
from extract_thinker.document_loader import DocumentLoaderTesseract

# Define classifications
classifications = [
    Classification(
        name="Driver License",
        description="This is a driver license",
        contract=DriverLicense,  # optional. Will be added to the prompt
    ),
    Classification(
        name="Invoice",
        description="This is an invoice",
        contract=InvoiceContract,  # optional. Will be added to the prompt
    ),
]

# Initialize extractor
tesseract_path = os.getenv("TESSERACT_PATH")
document_loader = DocumentLoaderTesseract(tesseract_path)
extractor = Extractor(document_loader)
extractor.load_llm("gpt-4o")

# Classify document
result = extractor.classify(INVOICE_FILE_PATH, classifications)
print(f"Document type: {result.name}, Confidence: {result.confidence}")
```

## Type Mapping with Contract

Adding contract structure to the classification improves accuracy:

```python
from typing import List
from extract_thinker.models.contract import Contract

class InvoiceContract(Contract):
    invoice_number: str
    invoice_date: str
    lines: List[LineItem]
    total_amount: float

class DriverLicense(Contract):
    name: str
    age: int
    license_number: str
```

The contract structure is automatically added to the prompt, helping the model understand the expected document structure.

## Classification Response

All classifications return a standardized response:

```python
from typing import Optional
from pydantic import BaseModel, Field

class ClassificationResponse(BaseModel):
    name: str
    confidence: Optional[int] = Field(
        description="From 1 to 10. 10 being the highest confidence",
        ge=1, 
        le=10
    )
```

## Best Practices

- Provide clear, distinctive descriptions for each classification
- Use contract structures when possible
- Consider using image classification for visual documents
- Monitor confidence scores
- Handle low-confidence cases appropriately

For more advanced classification techniques, see:
- [Mixture of Models (MoM)](mom.md)
- [Tree-Based Classification](tree.md)
- [Vision Classification](vision.md) 
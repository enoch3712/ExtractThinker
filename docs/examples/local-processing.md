# Local Processing with Ollama and Tesseract

This guide demonstrates how to process documents locally using Tesseract OCR and Ollama.

## Basic Setup

Here's how to use Tesseract with Ollama:

```python
from extract_thinker import Extractor, Contract, LLM, DocumentLoaderTesseract
from typing import List
from pydantic import Field

class InvoiceContract(Contract):
    invoice_number: str = Field("Invoice number")
    invoice_date: str = Field("Invoice date")
    total_amount: float = Field("Total amount")
    lines: List[LineItem] = Field("List of line items")

# Initialize Tesseract
extractor = Extractor()
extractor.load_document_loader(
    DocumentLoaderTesseract(os.getenv("TESSERACT_PATH"))
)

os.environ["API_BASE"] = "http://localhost:11434"

# Configure Ollama
extractor.load_llm("ollama/phi3")

# Process document
result = extractor.extract("invoice.pdf", InvoiceContract)
```

**Benefits - Privacy & Security**

- All processing done locally
- No data leaves your network
- Complete control over data
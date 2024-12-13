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
llm = LLM(
    "ollama/phi3",
)
extractor.load_llm(llm)

# Process document
result = extractor.extract("invoice.pdf", InvoiceContract)
```

## Advanced Configuration

Configure Tesseract for better accuracy:

```python
# Configure Tesseract with custom options
loader = DocumentLoaderTesseract(
    tesseract_path="/usr/local/bin/tesseract",
    config={
        "lang": "eng",  # Language
        "psm": 3,      # Page segmentation mode
        "oem": 3       # OCR Engine mode
    }
)

# Process with custom configuration
extractor.load_document_loader(loader)
result = extractor.extract("document.pdf", Contract)
```

## Benefits

1. **Privacy & Security**
   - All processing done locally
   - No data leaves your network
   - Complete control over data

2. **Cost Efficiency**
   - No usage fees
   - No API costs
   - Unlimited processing

3. **Customization**
   - Full control over models
   - Configurable processing
   - Custom training possible

## Best Practices

1. **Hardware Requirements**
   - Minimum 16GB RAM
   - GPU recommended for Ollama
   - SSD for faster processing

2. **Model Selection**
   - Phi-3 for English text
   - Llama for multilingual
   - Custom models as needed

3. **Error Handling**
   ```python
   try:
       result = extractor.extract(
           "document.pdf",
           Contract,
           config={
               "tesseract_timeout": 30,
               "ollama_timeout": 60
           }
       )
   except Exception as e:
       print(f"Processing error: {e}")
   ```
# Extract Thinker

ExtractThinker is the first Framework for LLMs just dedicated for Document Intelligence Processing (IDP).

<div align="center">
  <img src="../assets/extract-thinker-overview.png" alt="Extract Thinker Overview" width="50%">
</div>

🔍 **Document Processing**: Extract structured data from any document type (PDFs, images, etc.)

🤖 **LLM Integration**: Seamless integration with various LLM providers (OpenAI, Anthropic, local models)

⚡ **Async Support**: Process documents at scale with asynchronous batch processing

🧠 **Intelligent Classification**: Automatically classify and split documents with advanced ML

🎯 **Validation**: Built-in data validation through Pydantic contracts

## Installation

Install using pip:

```bash
pip install extract_thinker
```

## Quick Start

Here's a simple example that extracts invoice data from a PDF:

```python
from extract_thinker import Extractor, DocumentLoaderPyPdf, Contract

# Define what data you want to extract
class InvoiceContract(Contract):
    invoice_number: str
    invoice_date: str
    total_amount: float

# Initialize the extractor
extractor = Extractor()
extractor.load_document_loader(DocumentLoaderPyPdf())
extractor.load_llm("gpt-4")  # or any other supported model

# Extract data from your document
result = extractor.extract("invoice.pdf", InvoiceContract)

print(f"Invoice #{result.invoice_number}")
print(f"Date: {result.invoice_date}")
print(f"Total: ${result.total_amount}")
```

## Key Components

ExtractThinker is built around four main components:

- **DocumentLoader**: Handles document input (PDFs, images, etc.)
- **LLM**: Connects to language models for processing
- **Contract**: Defines the structure of data you want to extract
- **Extractor**: Orchestrates the entire extraction process

Check out our [advanced usage guide](./advanced-usage.md) for more complex scenarios like document classification, batch processing, and custom LLM integration.
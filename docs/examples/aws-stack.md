# AWS Textract with Claude Example

This guide demonstrates how to use AWS Textract combined with Claude for powerful document processing.

## Basic Setup

Here's how to combine AWS Textract's OCR capabilities with Claude:

```python
from extract_thinker import Extractor, Contract, LLM, DocumentLoaderTextract
from typing import List
from pydantic import Field

class InvoiceContract(Contract):
    invoice_number: str = Field("Invoice number")
    invoice_date: str = Field("Invoice date")
    total_amount: float = Field("Total amount")
    lines: List[LineItem] = Field("List of line items")

# Initialize AWS Textract
extractor = Extractor()
extractor.load_document_loader(
    DocumentLoaderTextract(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION")
    )
)

# Configure Claude
llm = LLM(
    "anthropic/claude-3-haiku-20240307",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)
extractor.load_llm(llm)

# Process document
result = extractor.extract("invoice.pdf", InvoiceContract)
```

## Cost Optimization

AWS Textract pricing:

- Basic text detection: $1.50 per 1,000 pages (first 1M pages/month)
                       $0.60 per 1,000 pages (over 1M pages/month)

- Tables: $15.00 per 1,000 pages (first 1M pages/month)
         $10.00 per 1,000 pages (over 1M pages/month)

Combined with Claude (via AWS Bedrock):

- Claude 3 Haiku: $0.00025 per 1K input tokens, $0.00125 per 1K output tokens

- Claude 3 Sonnet: $0.003 per 1K input tokens, $0.015 per 1K output tokens

Approximate cost per page (first 1M pages/month):
- Basic text only: $0.0015 per page ($1.50/1000)
- With tables: $0.0165 per page ($1.50/1000 + $15.00/1000)
- Plus Claude costs (varies by token length and model choice)
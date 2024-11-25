# Extractor

Extractor is the component that coordinates the extraction from documents. Contains a group of features for document processing like classify. Can be used alone or in group, inside of a Process.

<div align="center">
  <img src="../../assets/extractor.png" alt="Extractor">
</div>

## Basic Extraction

The simplest way to extract data is using the `Extractor` class with a defined contract:

```python
from extract_thinker import Extractor, DocumentLoaderPyPdf, Contract

class InvoiceContract(Contract):
	invoice_number: str
	invoice_date: str
	total_amount: float

#Initialize the extractor
extractor = Extractor()
extractor.load_document_loader(DocumentLoaderPyPdf())
extractor.load_llm("gpt-4o-mini") # or any other supported model

#Extract data from your document
result = extractor.extract("invoice.pdf", InvoiceContract)

print(f"Invoice #{result.invoice_number}")
print(f"Date: {result.invoice_date}")
print(f"Total: ${result.total_amount}")
```

---

## Choosing the Right Model

When performing extraction, selecting the appropriate model is crucial for balancing performance, accuracy, and cost:

- **GPT-4o-mini**: Best for basic text extraction tasks, similar to OCR. Cost-effective for high-volume processing.
- **GPT-4o**: Ideal for tasks requiring deeper understanding of document structure and content.
- **O1 and O1-mini**: Perfect for complex extraction requiring reasoning and calculations.

## Advanced Extraction with Vision

For documents that contain images or require visual understanding, you can enable vision capabilities:

```python
from extract_thinker import Extractor, Contract
from typing import List

class ChartData(Contract):
	title: str
	data_points: List[float]
	description: str

#Initialize with vision support
extractor = Extractor()
extractor.load_llm("gpt-4o")

#Extract with vision enabled
result = extractor.extract(
	"chart.png",
	ChartData,
	vision=True # Enable vision processing
)
```

> **Note:** When using vision capabilities, ensure your documents are high quality images or PDFs for optimal results.

## Adding Context to Extraction

You can provide additional context to help guide the extraction process:

```python
from extract_thinker import Extractor, Contract
class ResumeContract(Contract):
	name: str
	skills: List[str]
	experience: List[dict]

#Add context about the job requirements
job_description = {
	"role": "Software Engineer",
	"required_skills": ["Python", "AWS", "Docker"]
}

result = extractor.extract(
	"resume.pdf",
	ResumeContract,
	content=job_description # Add extra context
)
```

## Batch Processing

For handling large volumes of documents, ExtractThinker provides batch processing capabilities:

```python
from extract_thinker import Extractor, Contract

class ReceiptContract(Contract):
	store_name: str
	total_amount: float
	date: str

#Initialize batch processing
extractor = Extractor()
extractor.load_llm("gpt-4o-mini")

#Create batch job
batch_job = extractor.extract_batch(
	"receipts/.pdf",
	ReceiptContract
)

#Monitor status and get results
status = await batch_job.get_status()
results = await batch_job.get_result()
```

## Best Practices

**Model Selection**: Choose models based on task complexity:

   - Use `gpt-4o-mini` for basic text extraction
   - Use `gpt-4o` for structured data requiring context
   - Use `o1` models for complex reasoning tasks

**OCR Integration**: Combine OCR with LLM processing for better accuracy:

   - Use DocumentLoader for initial text extraction
   - Enable vision processing for complex layouts
   - Leverage both OCR text and image data when possible

**Batch Processing**: For large volumes:

   - Use `extract_batch` for cost-effective processing
   - Monitor job status with `get_status()`
   - Handle results asynchronously

## Error Handling

ExtractThinker provides robust error handling for extraction failures:

```python
from extract_thinker import Extractor, Contract
from extract_thinker.exceptions import ExtractionError
try:
result = extractor.extract("document.pdf", Contract)
except ExtractionError as e:
print(f"Extraction failed: {e}")
# Handle the error appropriately
```

For more examples and advanced usage, check out the [examples directory](examples/) in the repository.

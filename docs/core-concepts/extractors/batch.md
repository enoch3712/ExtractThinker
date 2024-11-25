# Batch Processing with Extractors

ExtractThinker provides powerful batch processing capabilities for handling large volumes of documents efficiently. This feature enables cost-effective processing when immediate response time is not critical.

## Basic Batch Processing

Here's how to use batch processing with the Extractor:

```python
from extract_thinker import Extractor, Contract

class InvoiceContract(Contract):
    invoice_number: str
    total_amount: float

# Initialize batch processing
extractor = Extractor()
extractor.load_llm("gpt-4o-mini")

# Create batch job
batch_job = extractor.extract_batch(
    "invoices/*.pdf",
    InvoiceContract
)

# Monitor status and get results
status = await batch_job.get_status()
results = await batch_job.get_result()
```

## Batch Job Status

Batch jobs can have the following statuses:

- `queued`: Job is waiting to be processed
- `processing`: Job is currently being processed
- `completed`: Job has finished successfully
- `failed`: Job encountered an error

## Cost Optimization

Batch processing can reduce costs by up to 50% compared to real-time processing. This is achieved by:

- Optimizing token usage across multiple documents
- Using more cost-effective models when time isn't critical
- Reducing API overhead through batched requests

## Best Practices

- Group similar documents together for batch processing
- Monitor job status to handle failures gracefully
- Use appropriate model sizes based on task complexity
- Implement retry logic for failed jobs

For more examples and advanced usage, check out the [examples directory](examples/) in the repository. 
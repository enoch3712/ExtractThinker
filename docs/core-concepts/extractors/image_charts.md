# Image and Chart Processing

ExtractThinker provides specialized capabilities for processing images and extracting data from charts using vision-enabled models. This guide covers how to effectively use these features.

<div align="center">
  <img src="../../../assets/chart_and_images.png" alt="Chart and Images">
</div>

## Basic Vision Processing

For documents containing images or requiring visual understanding:

```python
from extract_thinker import Extractor, Contract

class InvoiceContract(Contract):
    invoice_number: str
    invoice_date: str
    lines: List[LineItem]

# Initialize with vision support
extractor = Extractor()
extractor.load_llm("gpt-4o")

# Extract with vision enabled
result = extractor.extract(
    "invoice.pdf",
    InvoiceContract,
    vision=True  # Enable vision processing
)
```

## Chart Analysis

For extracting data from charts and graphs:

```python
from extract_thinker import Extractor, Contract
from typing import List, Literal

class Chart(Contract):
    classification: Literal['line', 'bar', 'pie']
    coordinates: List[XYCoordinate]
    description: str

class ChartWithContent(Contract):
    content: str  # Text content from the page
    chart: Chart  # Extracted chart data

# Initialize extractor for chart analysis
extractor = Extractor()
extractor.load_llm("gpt-4o")  # Required for chart analysis

# Extract chart data
result = extractor.extract(
    "chart.png",
    ChartWithContent,
    vision=True
)
```

## Model Selection for Visual Tasks

Different models are optimized for different visual tasks:

- **GPT-4o**: Required for vision tasks, chart analysis, and complex visual understanding
- **GPT-4o-mini**: Not suitable for vision tasks - use for text extraction only

## Best Practices

- Enable vision processing (`vision=True`) when working with images or charts
- Use GPT-4o or higher models for vision tasks
- Consider using a DocumentLoader in combination with vision for optimal results
- Ensure high-quality input images for best accuracy

## Limitations

- Vision processing requires GPT-4o or higher models
- Processing time may be longer for vision-enabled extraction
- Image quality significantly impacts extraction accuracy

For more examples and advanced usage, check out the [examples directory](examples/) in the repository. 
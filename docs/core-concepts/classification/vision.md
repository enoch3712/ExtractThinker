# Vision Classification

A document is not only text but also structure, color, and other numerous features that disappear when OCR is used. Vision classification leverages these visual elements to improve accuracy, particularly important for specific document types.

## Basic Usage

```python
from extract_thinker import Process, Classification
from extract_thinker.document_loader import DocumentLoaderTesseract

# Define classifications with example images
classifications = [
    Classification(
        name="Driver License",
        description="This is a driver license",
        contract=DriverLicense,
        image="path/to/example_license.png"  # Example image helps model understand
    ),
    Classification(
        name="Invoice",
        description="This is an invoice",
        contract=InvoiceContract,
        image="path/to/example_invoice.png"
    )
]

# Initialize process with vision-capable model
process = Process()
process.add_classify_extractor([[
    Extractor(DocumentLoaderTesseract(tesseract_path))
    .load_llm("gpt-4o")  # Vision-capable model
]])

# Classify with vision enabled
result = process.classify(
    "document.pdf",
    classifications,
    image=True  # Enable vision processing
)
```

## Benefits and Tradeoffs

### Benefits
- Better handling of document layouts
- Recognition of visual patterns and structures
- Improved accuracy for visually distinct documents
- Ability to understand non-textual elements

### Tradeoffs
- Higher cost due to image processing
- Larger context window requirements
- Longer processing times
- Higher token usage

## Model Selection

Different models offer varying capabilities for vision tasks:

- **GPT-4 Vision**: Supports low/high/auto quality settings (85 tokens for low)
- **Claude 3 Sonnet**: Full vision capabilities without quality options
- **Azure Phi-3 Vision**: Cost-effective alternative

## Best Practices

- Use compressed images when possible to reduce costs
- Provide high-quality example images for each classification
- Consider using a mix of vision and text-based classification
- Use appropriate image quality settings based on needs
- Cache vision results to avoid reprocessing

## Example with Multiple Models

```python
# Initialize extractors with different vision models
gpt4_vision = Extractor(document_loader)
gpt4_vision.load_llm("gpt-4-vision")

claude_vision = Extractor(document_loader)
claude_vision.load_llm("claude-3-sonnet")

phi3_vision = Extractor(document_loader)
phi3_vision.load_llm("phi-3-vision")

# Create process with vision models
process = Process()
process.add_classify_extractor([
    [phi3_vision],           # Cost-effective first attempt
    [claude_vision, gpt4_vision]  # More capable models if needed
])

# Classify with vision and consensus
result = process.classify(
    "document.pdf",
    classifications,
    strategy=ClassificationStrategy.CONSENSUS_WITH_THRESHOLD,
    threshold=9,
    image=True
)
```

For more examples and advanced usage, check out the [examples directory](examples/) in the repository. 
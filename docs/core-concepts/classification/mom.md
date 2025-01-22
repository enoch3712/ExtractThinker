# Mixture of Models (MoM)

The Mixture of Models (MoM) is a pattern that increases classification confidence by combining multiple models in parallel. This approach is particularly effective when using instructor for structured outputs.

## Basic Usage

```python
from extract_thinker import Process, Classification, ClassificationStrategy
from extract_thinker.document_loader import DocumentLoaderTesseract

# Define classifications
classifications = [
    Classification(
        name="Driver License",
        description="This is a driver license",
    ),
    Classification(
        name="Invoice",
        description="This is an invoice",
    ),
]

# Initialize document loader
tesseract_path = os.getenv("TESSERACT_PATH")
document_loader = DocumentLoaderTesseract(tesseract_path)

# Initialize multiple extractors with different models
gpt_35_extractor = Extractor(document_loader)
gpt_35_extractor.load_llm("gpt-3.5-turbo")

claude_extractor = Extractor(document_loader)
claude_extractor.load_llm("claude-3-haiku-20240307")

gpt4_extractor = Extractor(document_loader)
gpt4_extractor.load_llm("gpt-4o")

# Create process with multiple extractors
process = Process()
process.add_classify_extractor([
    [gpt_35_extractor, claude_3_haiku_extractor],  # First layer
    [gpt4_extractor],                              # Second layer
])

# Classify with consensus strategy
result = process.classify(
    "document.pdf",
    classifications,
    strategy=ClassificationStrategy.CONSENSUS_WITH_THRESHOLD,
    threshold=9
)
```

## Available Strategies

#### CONSENSUS
All models must agree on the classification:

```python
result = process.classify(
    "document.pdf",
    classifications,
    strategy=ClassificationStrategy.CONSENSUS
)
```

#### HIGHER_ORDER
Uses the result with the highest confidence score:

```python
result = process.classify(
    "document.pdf",
    classifications,
    strategy=ClassificationStrategy.HIGHER_ORDER
)
```

#### CONSENSUS_WITH_THRESHOLD
Requires both consensus and minimum confidence:

```python
result = process.classify(
    "document.pdf",
    classifications,
    strategy=ClassificationStrategy.CONSENSUS_WITH_THRESHOLD,
    threshold=9
)
```

## Best Practices

- Use smaller models in the first layer for cost efficiency
- Reserve larger models for cases where consensus isn't reached
- Set appropriate confidence thresholds based on your use case
- Consider using different model providers for better diversity
- Monitor and log classification results for each model

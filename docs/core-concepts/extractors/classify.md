# Classification

In document intelligence, classification is often the crucial first step. It sets the stage for subsequent processes like data extraction and analysis. ExtractThinker provides multiple strategies for accurate document classification.

## Basic Classification

The simplest way to classify documents:

```python
from extract_thinker import Extractor, Classification
from extract_thinker.document_loader import DocumentLoaderPyPdf

# Define classifications
classifications = [
    Classification(
        name="Invoice",
        description="This is an invoice document",
        contract=InvoiceContract  # Optional: adds structure to prompt
    ),
    Classification(
        name="Driver License",
        description="This is a driver license document",
        contract=DriverLicense
    )
]

# Initialize extractor
document_loader = DocumentLoaderPyPdf()
extractor = Extractor(document_loader)
extractor.load_llm("groq/llama-3.1-70b-versatile")

# Classify document
result = extractor.classify("document.pdf", classifications)
print(f"Document type: {result.name}")
```

## Mixture of Models (MoM)

For higher accuracy, you can use multiple models in parallel with different strategies:

```python
from extract_thinker import Process, ClassificationStrategy

# Initialize extractors with different models
gpt_35_extractor = Extractor(document_loader)
gpt_35_extractor.load_llm("gpt-3.5-turbo")

claude_extractor = Extractor(document_loader)
claude_extractor.load_llm("claude-3-haiku-20240307")

gpt4_extractor = Extractor(document_loader)
gpt4_extractor.load_llm("gpt-4o")

# Create process with multiple extractors
process = Process()
process.add_classify_extractor([
    [gpt_35_extractor, claude_extractor],  # First layer
    [gpt4_extractor]                       # Second layer
])

# Classify with consensus strategy
result = process.classify(
    "document.pdf", 
    classifications,
    strategy=ClassificationStrategy.CONSENSUS_WITH_THRESHOLD,
    threshold=9
)
```

## Tree-Based Classification

For handling many similar document types, use hierarchical classification:

```python
from extract_thinker import ClassificationNode, ClassificationTree

# Define classification tree
financial_docs = ClassificationNode(
    classification=Classification(
        name="Financial Documents",
        description="This is a financial document",
        contract=FinancialContract
    ),
    children=[
        ClassificationNode(
            classification=Classification(
                name="Invoice",
                description="This is an invoice",
                contract=InvoiceContract
            )
        ),
        ClassificationNode(
            classification=Classification(
                name="Credit Note",
                description="This is a credit note",
                contract=CreditNoteContract
            )
        )
    ]
)

# Create tree
classification_tree = ClassificationTree(
    nodes=[financial_docs]
)

# Classify using tree
result = process.classify(
    "document.pdf", 
    classification_tree, 
    threshold=0.95
)
```

## Classification Strategies

ExtractThinker supports three main classification strategies:

- **CONSENSUS**: All models must agree on the classification
- **HIGHER_ORDER**: Uses the result with highest confidence
- **CONSENSUS_WITH_THRESHOLD**: Requires consensus and minimum confidence

## Best Practices

- Use contract structures to improve accuracy
- Consider image-based classification for visual documents
- Implement tree-based classification for many similar documents
- Use multiple models for critical classifications
- Set appropriate confidence thresholds
- Monitor and log classification results

For more examples and advanced usage, check out the [examples directory](examples/) in the repository. 
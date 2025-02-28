# Field Comparison Types

When evaluating extraction results, different fields may require different comparison methods. For example:

- **ID fields** (like invoice numbers) typically require exact matching
- **Text descriptions** might benefit from semantic similarity comparison
- **Numeric values** could use tolerance-based comparison
- **Notes or comments** might allow for fuzzy matching

ExtractThinker's evaluation framework supports multiple comparison methods to address these different requirements.

## Available Comparison Types

| Comparison Type | Description | Best For |
|-----------------|-------------|----------|
| `EXACT` | Perfect string/value match (default) | IDs, codes, dates, categorical values |
| `FUZZY` | Approximate string matching using Levenshtein distance | Text with potential minor variations |
| `SEMANTIC` | Semantic similarity using embeddings | Descriptions, summaries, longer text |
| `NUMERIC` | Numeric comparison with percentage tolerance | Amounts, quantities, measurements |
| `CUSTOM` | Custom comparison function | Complex or domain-specific comparisons |

## Basic Usage

```python
from extract_thinker import Extractor, Contract
from extract_thinker.eval import Evaluator, FileSystemDataset, ComparisonType

# Define your contract
class InvoiceContract(Contract):
    invoice_number: str  # Needs exact matching
    description: str     # Can use semantic similarity
    total_amount: float  # Can use numeric tolerance

# Initialize your extractor
extractor = Extractor()
extractor.load_llm("gpt-4o")

# Create a dataset
dataset = FileSystemDataset(
    documents_dir="./test_invoices/",
    labels_path="./test_invoices/labels.json",
    name="Invoice Test Set"
)

# Set up evaluator with different field comparison types
evaluator = Evaluator(
    extractor=extractor,
    response_model=InvoiceContract,
    field_comparisons={
        "invoice_number": ComparisonType.EXACT,  # Must match exactly
        "description": ComparisonType.SEMANTIC,  # Compare meaning
        "total_amount": ComparisonType.NUMERIC   # Allow small % difference
    }
)

# Run evaluation
report = evaluator.evaluate(dataset)
```

## Configuring Comparison Parameters

Each comparison type has configurable parameters:

```python
# Configure thresholds for semantic similarity (description should be at least 80% similar)
evaluator.set_field_comparison(
    "description",
    ComparisonType.SEMANTIC,
    similarity_threshold=0.8
)

# Configure tolerance for numeric fields (total_amount can be within 2% of expected)
evaluator.set_field_comparison(
    "total_amount",
    ComparisonType.NUMERIC,
    numeric_tolerance=0.02
)
```

## Custom Comparison Functions

For specialized comparisons, you can define custom comparison functions:

```python
def compare_dates(expected, predicted):
    """Custom date comparison that handles different date formats."""
    from datetime import datetime
    # Try to parse both as dates
    try:
        expected_date = datetime.strptime(expected, "%Y-%m-%d")
        # Try different formats for predicted
        for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%B %d, %Y"]:
            try:
                predicted_date = datetime.strptime(predicted, fmt)
                return expected_date == predicted_date
            except ValueError:
                continue
        return False
    except ValueError:
        return expected == predicted

# Set custom comparison
evaluator.set_field_comparison(
    "invoice_date",
    ComparisonType.CUSTOM,
    custom_comparator=compare_dates
)
```

## Results Interpretation

The evaluation report will show which comparison type was used for each field:

```
=== Field-Level Metrics ===
invoice_number (comparison: exact):
  Precision: 98.00%
  Recall: 98.00%
  F1 Score: 98.00%
  Accuracy: 98.00%
description (comparison: semantic):
  Precision: 92.00%
  Recall: 92.00%
  F1 Score: 92.00%
  Accuracy: 92.00%
total_amount (comparison: numeric):
  Precision: 96.00%
  Recall: 96.00%
  F1 Score: 96.00%
  Accuracy: 96.00%
```

## Best Practices

- Use `EXACT` for fields where precise matching is critical (IDs, codes)
- Use `SEMANTIC` for long-form text that may vary in wording but should convey the same meaning
- Use `NUMERIC` for financial data, allowing for small rounding differences
- Use `FUZZY` for fields that may contain typos or minor variations
- Configure thresholds based on your application's tolerance for errors

## Advanced Usage

### Field Comparison Types

You can configure different comparison methods for different fields:

```python
from extract_thinker.eval import ComparisonType

evaluator = Evaluator(
    extractor=extractor,
    response_model=InvoiceContract,
    field_comparisons={
        "invoice_number": ComparisonType.EXACT,  # Exact match required
        "description": ComparisonType.SEMANTIC,  # Semantic similarity
        "total_amount": ComparisonType.NUMERIC   # Allows percentage tolerance
    }
)
```

For more details, see [Field Comparison Types](field-comparison.md).
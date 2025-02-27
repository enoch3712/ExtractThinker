# ExtractThinker Evaluation Framework

## Overview

The evaluation framework helps you:
- Measure extraction accuracy at both field and document levels
- Track schema validation success rates
- Monitor execution times
- Generate comprehensive reports with detailed metrics
- Compare performance across different models or datasets

## Basic Usage

Here's how to set up a basic evaluation:

```python
from extract_thinker import Extractor, Contract
from extract_thinker.eval import Evaluator, FileSystemDataset
from typing import List

# Define your contract
class InvoiceContract(Contract):
    invoice_number: str
    date: str
    total_amount: float
    line_items: List[dict]

# Initialize your extractor
extractor = Extractor()
extractor.load_llm("gpt-4o")

# Create a dataset
dataset = FileSystemDataset(
    documents_dir="./test_invoices/",
    labels_path="./test_invoices/labels.json",
    name="Invoice Test Set"
)

# Set up evaluator
evaluator = Evaluator(
    extractor=extractor,
    response_model=InvoiceContract
)

# Run evaluation
report = evaluator.evaluate(dataset)

# Print summary and save detailed report
report.print_summary()
evaluator.save_report(report, "evaluation_results.json")
```

## Using the Command Line Interface

ExtractThinker includes a CLI for running evaluations from configuration files:

```bash
extract_thinker-eval --config eval_config.json --output results.json
```

Example configuration file:

```json
{
  "evaluation_name": "Invoice Extraction Test",
  "dataset_name": "Invoice Dataset",
  "contract_path": "./contracts/invoice_contract.py",
  "documents_dir": "./test_invoices/",
  "labels_path": "./test_invoices/labels.json",
  "file_pattern": "*.pdf",
  "llm": {
    "model": "gpt-4o",
    "params": {
      "temperature": 0.0
    }
  },
  "vision": false,
  "skip_failures": false
}
```

## Available Metrics

The evaluation framework provides several metrics to assess model performance:

| Metric | Description |
|--------|-------------|
| Document Accuracy | Percentage of documents with all fields correctly extracted |
| Schema Validation Rate | Percentage of documents that produce valid schema outputs |
| Field Precision | Proportion of extracted field values that are correct |
| Field Recall | Proportion of expected field values that are correctly extracted |
| Field F1 Score | Harmonic mean of precision and recall |
| Execution Time | Average time taken to extract information from a document |

## Creating Evaluation Datasets

ExtractThinker supports filesystem-based evaluation datasets:

```python
from extract_thinker.eval import FileSystemDataset

# Create a dataset from files on disk
dataset = FileSystemDataset(
    documents_dir="./documents/",
    labels_path="./labels.json",
    name="Custom Dataset",
    file_pattern="*.pdf"  # Optional glob pattern
)
```

The labels file should be a JSON file mapping document filenames to expected values:

```json
{
  "invoice1.pdf": {
    "invoice_number": "INV-2024-001",
    "date": "2024-05-15",
    "total_amount": 1250.50,
    "line_items": [
      {"description": "Service A", "amount": 1000.00},
      {"description": "Service B", "amount": 250.50}
    ]
  },
  "invoice2.pdf": {
    ...
  }
}
```

## Interpreting Results

The evaluation report provides both overall metrics and field-level performance:

```
=== Invoice Extraction Test ===
Dataset: Invoice Dataset
Model: gpt-4o
Timestamp: 2024-06-01T12:34:56.789012

=== Overall Metrics ===
Documents tested: 50
Document accuracy: 85.00%
Schema validation rate: 98.00%
Average precision: 92.50%
Average recall: 90.00%
Average F1 score: 91.23%
Average execution time: 2.45s

=== Field-Level Metrics ===
invoice_number:
  Precision: 98.00%
  Recall: 98.00%
  F1 Score: 98.00%
  Accuracy: 98.00%
date:
  Precision: 94.00%
  Recall: 94.00%
  F1 Score: 94.00%
  Accuracy: 94.00%
...
```

## Advanced Usage

### Vision Support

For documents that require visual processing:

```python
evaluator = Evaluator(
    extractor=extractor,
    response_model=InvoiceContract,
    vision=True  # Enable vision mode
)
```

### Additional Content

You can provide additional context or instructions for extraction:

```python
evaluator = Evaluator(
    extractor=extractor,
    response_model=InvoiceContract,
    content="Focus on the header section of the invoice for the invoice number and date."
)
```

### Skip Failed Documents

To continue evaluation even when schema validation fails:

```python
report = evaluator.evaluate(
    dataset=dataset,
    skip_failures=True
)
```

## Best Practices

- Create diverse datasets that cover a range of document variations
- Use consistent file formats and naming conventions in your datasets
- Run evaluations on different model configurations to find optimal settings
- Monitor field-level metrics to identify specific areas for improvement
- Create separate test sets for different document types
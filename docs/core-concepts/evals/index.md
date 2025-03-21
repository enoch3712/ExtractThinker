# Evaluation Framework <span class="beta-badge">🧪 In Beta</span>

The evaluation framework helps measure the performance and reliability of your extraction models across different document types.

## Overview

ExtractThinker's evaluation system provides comprehensive metrics to:

* Measure extraction accuracy at both field and document levels
* Track schema validation success rates
* Monitor execution times
* Detect potential hallucinations in extracted data
* Track token usage and associated costs
* Compare performance across different models or datasets

## Required Components

To use the evaluation framework, you'll need:

* An initialized `Extractor` instance
* A `Contract` class that defines your extraction schema
* A dataset containing documents and their expected outputs

## Basic Usage

Here's how to set up and run a basic evaluation:

```python
from extract_thinker import Extractor, Contract
from extract_thinker.eval import Evaluator, FileSystemDataset
from typing import List

# 1. Define your contract class
class InvoiceContract(Contract):
    invoice_number: str
    date: str
    total_amount: float
    line_items: List[dict]

# 2. Initialize your extractor
extractor = Extractor()
extractor.load_llm("gpt-4o")

# 3. Create a dataset
dataset = FileSystemDataset(
    documents_dir="./test_invoices/",
    labels_path="./test_invoices/labels.json",
    name="Invoice Test Set"
)

# 4. Set up evaluator
evaluator = Evaluator(
    extractor=extractor,
    response_model=InvoiceContract
)

# 5. Run evaluation
report = evaluator.evaluate(dataset)

# 6. Print summary and save detailed report
report.print_summary()
evaluator.save_report(report, "evaluation_results.json")
```

> **Tip:** For consistent evaluations, use a temperature of 0.0 in your model configuration to ensure deterministic outputs.

!!! tip "💡 Model Temperature"
    For consistent evaluations, use a temperature of 0.0 in your model configuration to ensure deterministic outputs.

## Command Line Interface

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

ExtractThinker captures several key metrics during evaluation:

| Metric Type | Description | Use Case |
|-------------|-------------|----------|
| Field-level | Precision, recall, F1 scores for each field | Identify problematic fields |
| Document-level | Overall accuracy across all documents | General model performance |
| Schema validation | Success rate of schema validation | Data structure correctness |
| Execution time | Average and per-document processing time | Performance optimization |
| Hallucination | Detection of fabricated information | Trust and reliability |
| Cost | Token usage and associated costs | Budget optimization |

### Sample Output

```
=== Invoice Extraction Evaluation ===
Dataset: Invoice Test Set
Model: gpt-4o
Timestamp: 2023-08-15T14:30:45

=== Overall Metrics ===
Documents tested: 50
Document accuracy: 92.00%
Schema validation rate: 96.00%
Average precision: 95.40%
Average recall: 94.80%
Average F1 score: 95.10%
Average execution time: 2.34s

=== Field-Level Metrics ===
invoice_number (comparison: exact):
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

## Evaluation Features

ExtractThinker offers several specialized evaluation capabilities:

### Field Comparison Types

Different fields may require different comparison methods:

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

[Learn more about field comparison types →](field-comparison.md)

### Teacher-Student Evaluation

Benchmark your model against a more capable "teacher" model:

```python
from extract_thinker.eval import TeacherStudentEvaluator

evaluator = TeacherStudentEvaluator(
    student_extractor=student_extractor,
    teacher_extractor=teacher_extractor,
    response_model=InvoiceContract
)
```

[Learn more about teacher-student evaluation →](teacher-student.md)

### Hallucination Detection

Identify potentially hallucinated content:

```python
evaluator = Evaluator(
    extractor=extractor,
    response_model=InvoiceContract,
    detect_hallucinations=True
)
```

[Learn more about hallucination detection →](hallucination-detection.md)

### Cost Tracking

Monitor token usage and costs:

```python
evaluator = Evaluator(
    extractor=extractor,
    response_model=InvoiceContract,
    track_costs=True
)
```

[Learn more about cost tracking →](cost-tracking.md)

## Best Practices


* **Dataset diversity**: Include a wide range of document variations in your test set
* **Consistent formatting**: Use consistent file formats and naming conventions
* **Benchmark different models**: Run evaluations on different model configurations
* **Field-level analysis**: Monitor field-level metrics to identify specific areas for improvement
* **Specialized test sets**: Create separate test sets for different document types
* **Hallucination checks**: Enable hallucination detection for critical applications
* **Cost optimization**: Track costs to optimize the performance/price ratio
* **Version control**: Keep evaluation datasets under version control to track improvements over time

## Advanced Configuration

For more complex evaluation needs:

```python
# Advanced evaluator setup with multiple features
evaluator = Evaluator(
    extractor=extractor,
    response_model=InvoiceContract,
    vision=True,  # Enable vision mode for image-based documents
    content="Focus on the header section for invoice number and date.",
    field_comparisons={
        "invoice_number": ComparisonType.EXACT,
        "description": ComparisonType.SEMANTIC
    },
    detect_hallucinations=True,
    track_costs=True
)

# Run evaluation with special options
report = evaluator.evaluate(
    dataset=dataset,
    evaluation_name="Comprehensive Invoice Evaluation",
    skip_failures=True  # Continue even when schema validation fails
)
```
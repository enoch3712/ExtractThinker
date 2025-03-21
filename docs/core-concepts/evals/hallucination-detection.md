# Hallucination Detection <span class="beta-badge">🧪 In Beta</span>

## Overview

When extracting information from documents, language models sometimes "hallucinate" content by generating information that isn't actually present in the source document. ExtractThinker's hallucination detection helps identify these cases.

## Basic Usage

To enable hallucination detection in your evaluations:

```python
from extract_thinker import Extractor, Contract
from extract_thinker.eval import Evaluator, FileSystemDataset

# Initialize your extractor
extractor = Extractor()
extractor.load_llm("gpt-4o")

# Create evaluator with hallucination detection enabled
evaluator = Evaluator(
    extractor=extractor,
    response_model=YourContract,
    detect_hallucinations=True  # Enable hallucination detection
)

# Run evaluation
report = evaluator.evaluate(dataset)
```

## Command Line Usage

You can also enable hallucination detection through the CLI:

```bash
extract_thinker-eval --config eval_config.json --output results.json --detect-hallucinations
```

Or in your config file:

```json
{
  "evaluation_name": "Invoice Extraction Test",
  "dataset_name": "Invoice Dataset",
  "contract_path": "./contracts/invoice_contract.py",
  "documents_dir": "./test_invoices/",
  "labels_path": "./test_invoices/labels.json",
  "detect_hallucinations": true,
  "llm": {
    "model": "gpt-4o"
  }
}
```

## How It Works

The hallucination detector compares extracted field values against the source document text to determine whether the information could reasonably have been derived from the document.

For each field, a hallucination score between 0.0 and 1.0 is calculated:

- **0.0 - 0.3**: Content is present in the document
- **0.3 - 0.7**: Content might be partially inferred from the document
- **0.7 - 1.0**: Content appears to be hallucinated

The detector uses two strategies:

1. **Heuristic detection**: Uses pattern matching and text similarity to check if extracted values appear in the document
2. **LLM-assisted detection**: Uses an LLM to determine if extracted information could reasonably be inferred from the document

## Interpreting Results

Hallucination scores appear in the evaluation report:

```
=== Hallucination Metrics ===
Average hallucination score: 0.32
Fields with potential hallucinations: 3

=== Field-Level Metrics ===
invoice_number (comparison: exact):
  Precision: 96.00%
  Recall: 96.00%
  F1 Score: 96.00%
  Accuracy: 96.00%
  Hallucination score: 0.05
description (comparison: semantic):
  Precision: 89.00%
  Recall: 89.00%
  F1 Score: 89.00%
  Accuracy: 89.00%
  Hallucination score: 0.78
```

The hallucination data is also available programmatically:

```python
# Access hallucination data for specific documents
for result in report.results:
    if "hallucination_results" in result:
        hallucination_data = result["hallucination_results"]
        print(f"Document {result['doc_id']} overall hallucination score: {hallucination_data['overall_score']}")
        
        # Field-specific hallucination scores
        for field_name, score in hallucination_data["field_scores"].items():
            print(f"  {field_name}: {score}")
            
        # Detailed reasoning for hallucination detection
        for detail in hallucination_data["detailed_results"]:
            print(f"  {detail['field_name']}: {detail['hallucination_score']} - {detail['reasoning']}")
```

## Advanced Usage

You can customize the hallucination detector by directly working with the `HallucinationDetector` class:

```python
from extract_thinker.eval import HallucinationDetector

# Create a custom detector with a different threshold
detector = HallucinationDetector(
    llm=extractor.llm,
    threshold=0.8  # More tolerant threshold (default is 0.7)
)

# Run detection on extracted data
results = detector.detect_hallucinations(
    extracted_data=extracted_data,
    document_text=document_text
)
```

## Best Practices

1. **Always enable for sensitive data extraction**: For financial, legal, or medical documents
2. **Review high hallucination scores**: Fields with scores above 0.7 should be manually verified
3. **Use with LLM-enabled detection**: Providing an LLM for detection offers more nuanced results
4. **Compare with ground truth**: High hallucination scores combined with incorrect extraction indicate model confabulation 
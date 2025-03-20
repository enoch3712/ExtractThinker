# Cost Tracking <span class="beta-badge">🧪 In Beta</span>
## Overview

ExtractThinker provides built-in cost tracking for evaluations, helping you monitor token usage and associated costs when using various LLM models.

## Basic Usage

To enable cost tracking in your evaluations:

```python
from extract_thinker import Extractor, Contract
from extract_thinker.eval import Evaluator, FileSystemDataset

# Initialize your extractor
extractor = Extractor()
extractor.load_llm("gpt-4o")

# Create evaluator with cost tracking enabled
evaluator = Evaluator(
    extractor=extractor,
    response_model=YourContract,
    track_costs=True  # Enable cost tracking
)

# Run evaluation
report = evaluator.evaluate(dataset)
```

## Command Line Usage

You can also enable cost tracking through the CLI:

```bash
extract_thinker-eval --config eval_config.json --output results.json --track-costs
```

Or in your config file:

```json
{
  "evaluation_name": "Invoice Extraction Test",
  "dataset_name": "Invoice Dataset",
  "contract_path": "./contracts/invoice_contract.py",
  "documents_dir": "./test_invoices/",
  "labels_path": "./test_invoices/labels.json",
  "track_costs": true,
  "llm": {
    "model": "gpt-4o"
  }
}
```

## How It Works

Cost tracking leverages LiteLLM's built-in cost calculation features to:

1. Count input and output tokens for each extraction
2. Calculate costs based on current model pricing
3. Aggregate metrics across all evaluated documents

## Interpreting Results

Cost metrics appear in the evaluation report:

```
=== Cost Metrics ===
Total cost: $2.4768
Average cost per document: $0.0495
Total tokens: 123,840
  - Input tokens: 98,450
  - Output tokens: 25,390
```

The cost data is also available programmatically:

```python
# Access overall cost metrics
total_cost = report.metrics["total_cost"]
average_cost = report.metrics["average_cost"]
total_tokens = report.metrics["total_tokens"]

# Access document-specific costs
for result in report.results:
    doc_id = result["doc_id"]
    doc_tokens = result["tokens"]
    doc_cost = result["cost"]
    
    print(f"Document: {doc_id}")
    print(f"  Cost: ${doc_cost:.4f}")
    print(f"  Input tokens: {doc_tokens['input']}")
    print(f"  Output tokens: {doc_tokens['output']}")
    print(f"  Total tokens: {doc_tokens['total']}")
```

## Cost-Benefit Analysis

Cost tracking is particularly useful for:

1. **Model comparison**: Understand the cost-accuracy tradeoffs between different models
2. **Optimization**: Identify expensive documents that might need prompt optimization
3. **Budgeting**: Estimate production deployment costs based on evaluation results
4. **ROI calculation**: Calculate return on investment by comparing accuracy improvements to increased costs

## Teacher-Student Integration

Cost tracking works seamlessly with the teacher-student evaluation approach to help quantify the cost-benefit relationship of using more capable models:

```python
from extract_thinker.eval import TeacherStudentEvaluator

# Set up teacher-student evaluator with cost tracking
evaluator = TeacherStudentEvaluator(
    student_extractor=student_extractor,
    teacher_extractor=teacher_extractor,
    response_model=InvoiceContract,
    track_costs=True  # Enable cost tracking
)

# Run evaluation
report = evaluator.evaluate(dataset)

# The report will include cost differences between teacher and student models
student_cost = report.metrics["student_average_cost"]
teacher_cost = report.metrics["teacher_average_cost"]
cost_ratio = teacher_cost / student_cost

print(f"Cost ratio (teacher/student): {cost_ratio:.2f}x")
print(f"Accuracy improvement: {report.metrics['document_accuracy_improvement']:.2f}%")
```

This helps answer questions like "Is a 15% accuracy improvement worth a 3x cost increase?"

## Supported Models

Cost tracking works with all models supported by LiteLLM, including:
- OpenAI models (GPT-3.5, GPT-4, etc.)
- Claude models (Claude 3 Opus, Sonnet, etc.)
- Mistral models
- Most other major LLM providers

## Limitations

- Costs are estimated based on current pricing and may not reflect custom pricing arrangements
- For some models, costs may be approximate if token counting methods vary
- Document loading/preprocessing costs are not included 
# Teacher-Student Evaluation

The teacher-student approach allows you to benchmark your extractor against a more capable "teacher" model to identify performance gaps and potential areas for improvement.

This approach is particularly useful for:
- Understanding the performance ceiling with better models
- Identifying fields that benefit most from model upgrades
- Quantifying the cost-performance tradeoff between different models
- Creating a baseline for progressive model improvement

## Basic Setup

```python
from extract_thinker import Extractor, Contract
from extract_thinker.eval import TeacherStudentEvaluator, FileSystemDataset
from extract_thinker.document_loader import DocumentLoaderPyPdf, DocumentLoaderAWSTextract

# Define your contract
class InvoiceContract(Contract):
    invoice_number: str
    date: str
    total_amount: float
    line_items: List[dict]

# Initialize "student" extractor with standard configuration
student_extractor = Extractor()
student_extractor.load_document_loader(DocumentLoaderPyPdf())
student_extractor.load_llm("gpt-4o-mini") # More affordable model

# Initialize "teacher" extractor with superior configuration
teacher_extractor = Extractor()
teacher_extractor.load_document_loader(DocumentLoaderAWSTextract())
teacher_extractor.load_llm("gpt-4o") # More capable model

# Create dataset
dataset = FileSystemDataset(
    documents_dir="./test_invoices/",
    labels_path="./test_invoices/labels.json",
    name="Invoice Test Set"
)

# Set up teacher-student evaluator
evaluator = TeacherStudentEvaluator(
    student_extractor=student_extractor,
    teacher_extractor=teacher_extractor,
    response_model=InvoiceContract
)

# Run comparative evaluation
report = evaluator.evaluate(dataset)

# Print comparative summary
report.print_summary()
```

## Interpreting Comparative Results

The evaluation report provides side-by-side metrics for both models and the improvement percentages:

```
=== Teacher-Student Evaluation ===
Dataset: Invoice Test Set
Model(s): Student: gpt-4o-mini, Teacher: gpt-4o
Timestamp: 2024-06-01T12:34:56.789012

=== Student Model Metrics ===
Documents tested: 50
Document accuracy: 75.00%
Schema validation rate: 92.00%
Average precision: 85.50%
Average recall: 82.00%
Average F1 score: 83.70%
Average execution time: 1.85s

=== Teacher Model Metrics ===
Document accuracy: 94.00%
Schema validation rate: 100.00%
Average precision: 96.50%
Average recall: 95.00%
Average F1 score: 95.74%
Average execution time: 3.25s

=== Comparison Metrics ===
Document accuracy improvement: 25.33%
Execution time ratio (teacher/student): 1.76x

=== Field-Level Improvements ===
invoice_number:
  Student F1: 92.00%
  Teacher F1: 98.00%
  Improvement: 6.52%
date:
  Student F1: 88.00%
  Teacher F1: 96.00%
  Improvement: 9.09%
total_amount:
  Student F1: 78.00%
  Teacher F1: 94.00%
  Improvement: 20.51%
...
```

## Advanced Configuration

You can configure different settings for the student and teacher models:

```python
evaluator = TeacherStudentEvaluator(
    student_extractor=student_extractor,
    teacher_extractor=teacher_extractor,
    response_model=InvoiceContract,
    student_vision=False,
    teacher_vision=True, # Enable vision only for teacher
    student_content="Extract the basic invoice details.",
    teacher_content="Extract all invoice details with high precision."
)
```

## Cost-Benefit Analysis

The teacher-student approach helps you make informed decisions about model selection by quantifying the performance gains relative to the additional cost and processing time of more capable models.

Consider the following when interpreting results:
- Is the accuracy improvement worth the increased cost?
- Are there specific fields that benefit more from the teacher model?
- Would a different document loader provide better results without increasing model costs?
- Can you tailor prompts to close the gap between student and teacher performance?
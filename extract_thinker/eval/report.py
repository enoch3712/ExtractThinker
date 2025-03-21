from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class EvaluationReport(BaseModel):
    """
    Structured report for evaluation results with teacher-student comparison support.
    """
    
    evaluation_name: str = Field(..., description="Name of the evaluation")
    dataset: str = Field(..., description="Name of the dataset used")
    model: str = Field(..., description="Name of the model(s) used for extraction")
    timestamp: str = Field(..., description="Timestamp of the evaluation")
    documents_evaluated: int = Field(..., description="Number of documents evaluated")
    
    metrics: Dict[str, Any] = Field(
        ...,
        description="Overall metrics including accuracy, precision, recall, etc."
    )
    
    field_metrics: Dict[str, Dict[str, float]] = Field(
        ...,
        description="Student metrics broken down by field"
    )
    
    teacher_field_metrics: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="Teacher metrics broken down by field (if present)"
    )
    
    field_improvements: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="Field-by-field improvement metrics (if teacher present)"
    )
    
    comparison_configs: Optional[Dict[str, Dict[str, Any]]] = Field(
        None,
        description="Field comparison configurations used for evaluation"
    )
    
    results: List[Dict[str, Any]] = Field(
        ...,
        description="Detailed results for each document from student model"
    )
    
    teacher_results: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Detailed results for each document from teacher model (if present)"
    )
    
    cost_metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="Token usage and cost metrics if cost tracking was enabled"
    )
    
    hallucination_metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="Hallucination detection metrics if enabled"
    )
    
    def print_summary(self):
        """Print a human-readable summary of the evaluation results."""
        print(f"\n=== {self.evaluation_name} ===")
        print(f"Dataset: {self.dataset}")
        print(f"Model(s): {self.model}")
        print(f"Timestamp: {self.timestamp}")
        
        # Check if this is a teacher-student comparison report
        is_teacher_student = "teacher_document_accuracy" in self.metrics
        
        if is_teacher_student:
            self._print_teacher_student_summary()
        else:
            self._print_standard_summary()
    
    def _print_standard_summary(self):
        """Print summary for standard evaluation."""
        print("\n=== Overall Metrics ===")
        print(f"Documents tested: {self.metrics['documents_tested']}")
        print(f"Document accuracy: {self.metrics['overall_document_accuracy']:.2%}")
        print(f"Schema validation rate: {self.metrics['schema_validation_rate']:.2%}")
        print(f"Average precision: {self.metrics['average_precision']:.2%}")
        print(f"Average recall: {self.metrics['average_recall']:.2%}")
        print(f"Average F1 score: {self.metrics['average_f1']:.2%}")
        print(f"Average execution time: {self.metrics['average_execution_time_s']:.2f}s")
        
        # Print cost metrics if available
        if "total_cost" in self.metrics:
            print("\n=== Cost Metrics ===")
            print(f"Total cost: ${self.metrics['total_cost']:.4f}")
            print(f"Average cost per document: ${self.metrics['average_cost']:.4f}")
            print(f"Total tokens: {self.metrics['total_tokens']}")
            print(f"  - Input tokens: {self.metrics['total_input_tokens']}")
            print(f"  - Output tokens: {self.metrics['total_output_tokens']}")
        
        # Print hallucination metrics if available
        if self.hallucination_metrics:
            print("\n=== Hallucination Metrics ===")
            print(f"Average hallucination score: {self.hallucination_metrics['average_score']:.2f}")
            print(f"Fields with potential hallucinations: {self.hallucination_metrics['hallucinated_field_count']}")
        
        print("\n=== Field-Level Metrics ===")
        for field, metrics in self.field_metrics.items():
            comparison_type = "exact"
            if self.comparison_configs and field in self.comparison_configs:
                comparison_type = self.comparison_configs[field].get("comparison_type", "exact")
                
            print(f"{field} (comparison: {comparison_type}):")
            print(f"  Precision: {metrics['precision']:.2%}")
            print(f"  Recall: {metrics['recall']:.2%}")
            print(f"  F1 Score: {metrics['f1']:.2%}")
            print(f"  Accuracy: {metrics['accuracy']:.2%}")
            
            # Print hallucination score for field if available
            if (self.hallucination_metrics and 
                'field_scores' in self.hallucination_metrics and 
                field in self.hallucination_metrics['field_scores']):
                hall_score = self.hallucination_metrics['field_scores'][field]
                print(f"  Hallucination score: {hall_score:.2f}")
    
    def _print_teacher_student_summary(self):
        """Print summary for teacher-student comparison."""
        print("\n=== Student Model Metrics ===")
        print(f"Documents tested: {self.metrics['documents_tested']}")
        print(f"Document accuracy: {self.metrics['student_document_accuracy']:.2%}")
        print(f"Schema validation rate: {self.metrics['student_schema_validation_rate']:.2%}")
        print(f"Average precision: {self.metrics['student_average_precision']:.2%}")
        print(f"Average recall: {self.metrics['student_average_recall']:.2%}")
        print(f"Average F1 score: {self.metrics['student_average_f1']:.2%}")
        print(f"Average execution time: {self.metrics['student_average_execution_time_s']:.2f}s")
        
        print("\n=== Teacher Model Metrics ===")
        print(f"Document accuracy: {self.metrics['teacher_document_accuracy']:.2%}")
        print(f"Schema validation rate: {self.metrics['teacher_schema_validation_rate']:.2%}")
        print(f"Average precision: {self.metrics['teacher_average_precision']:.2%}")
        print(f"Average recall: {self.metrics['teacher_average_recall']:.2%}")
        print(f"Average F1 score: {self.metrics['teacher_average_f1']:.2%}")
        print(f"Average execution time: {self.metrics['teacher_average_execution_time_s']:.2f}s")
        
        print("\n=== Comparison Metrics ===")
        print(f"Document accuracy improvement: {self.metrics['document_accuracy_improvement']:.2f}%")
        print(f"Execution time ratio (teacher/student): {self.metrics['execution_time_ratio']:.2f}x")
        
        print("\n=== Field-Level Improvements ===")
        for field, improvements in self.field_improvements.items():
            comparison_type = "exact"
            if self.comparison_configs and field in self.comparison_configs:
                comparison_type = self.comparison_configs[field].get("comparison_type", "exact")
                
            print(f"{field} (comparison: {comparison_type}):")
            print(f"  Student F1: {improvements['student_f1']:.2%}")
            print(f"  Teacher F1: {improvements['teacher_f1']:.2%}")
            print(f"  Improvement: {improvements['improvement_pct']:.2f}%") 
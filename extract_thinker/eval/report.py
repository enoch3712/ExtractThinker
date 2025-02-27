from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class EvaluationReport(BaseModel):
    """
    Structured report for evaluation results.
    """
    
    evaluation_name: str = Field(..., description="Name of the evaluation")
    dataset: str = Field(..., description="Name of the dataset used")
    model: str = Field(..., description="Name of the model used for extraction")
    timestamp: str = Field(..., description="Timestamp of the evaluation")
    
    metrics: Dict[str, Any] = Field(
        ...,
        description="Overall metrics including accuracy, precision, recall, etc."
    )
    
    field_metrics: Dict[str, Dict[str, float]] = Field(
        ...,
        description="Metrics broken down by field"
    )
    
    results: List[Dict[str, Any]] = Field(
        ...,
        description="Detailed results for each document"
    )
    
    def print_summary(self):
        """Print a human-readable summary of the evaluation results."""
        print(f"\n=== {self.evaluation_name} ===")
        print(f"Dataset: {self.dataset}")
        print(f"Model: {self.model}")
        print(f"Timestamp: {self.timestamp}")
        print("\n=== Overall Metrics ===")
        print(f"Documents tested: {self.metrics['documents_tested']}")
        print(f"Document accuracy: {self.metrics['overall_document_accuracy']:.2%}")
        print(f"Schema validation rate: {self.metrics['schema_validation_rate']:.2%}")
        print(f"Average precision: {self.metrics['average_precision']:.2%}")
        print(f"Average recall: {self.metrics['average_recall']:.2%}")
        print(f"Average F1 score: {self.metrics['average_f1']:.2%}")
        print(f"Average execution time: {self.metrics['average_execution_time_s']:.2f}s")
        
        print("\n=== Field-Level Metrics ===")
        for field, metrics in self.field_metrics.items():
            print(f"{field}:")
            print(f"  Precision: {metrics['precision']:.2%}")
            print(f"  Recall: {metrics['recall']:.2%}")
            print(f"  F1 Score: {metrics['f1']:.2%}")
            print(f"  Accuracy: {metrics['accuracy']:.2%}") 
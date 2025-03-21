from typing import Dict, Any, Type, List
import statistics
from pydantic import BaseModel


class FieldMetrics:
    """
    Calculate and store field-level metrics like precision, recall, and F1.
    """
    
    def __init__(self, model_class: Type[BaseModel]):
        """
        Initialize field metrics calculator.
        
        Args:
            model_class: The Contract class to extract field names from
        """
        # Get field names from model
        self.field_names = list(model_class.__annotations__.keys())
        
        # Initialize metrics counters for each field
        self.reset()
    
    def reset(self):
        """Reset all metrics counters."""
        self.true_positives = {field: 0 for field in self.field_names}
        self.false_positives = {field: 0 for field in self.field_names}
        self.false_negatives = {field: 0 for field in self.field_names}
        self.total = {field: 0 for field in self.field_names}
    
    def update(self, field_name: str, correct: bool, present: bool):
        """
        Update metrics for a field.
        
        Args:
            field_name: Name of the field
            correct: Whether the field value was correct
            present: Whether the field was present in the prediction
        """
        self.total[field_name] += 1
        
        if correct:
            self.true_positives[field_name] += 1
        elif present:
            self.false_positives[field_name] += 1
        else:
            self.false_negatives[field_name] += 1
    
    def get_precision(self, field_name: str = None) -> float:
        """
        Calculate precision for a field or average across all fields.
        
        Args:
            field_name: Optional field name to calculate precision for
            
        Returns:
            float: Precision value between 0 and 1
        """
        if field_name:
            tp = self.true_positives[field_name]
            fp = self.false_positives[field_name]
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0
        else:
            # Average precision across all fields
            precisions = [self.get_precision(field) for field in self.field_names]
            return statistics.mean(precisions) if precisions else 0.0
    
    def get_recall(self, field_name: str = None) -> float:
        """
        Calculate recall for a field or average across all fields.
        
        Args:
            field_name: Optional field name to calculate recall for
            
        Returns:
            float: Recall value between 0 and 1
        """
        if field_name:
            tp = self.true_positives[field_name]
            fn = self.false_negatives[field_name]
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:
            # Average recall across all fields
            recalls = [self.get_recall(field) for field in self.field_names]
            return statistics.mean(recalls) if recalls else 0.0
    
    def get_f1(self, field_name: str = None) -> float:
        """
        Calculate F1 score for a field or average across all fields.
        
        Args:
            field_name: Optional field name to calculate F1 for
            
        Returns:
            float: F1 score between 0 and 1
        """
        if field_name:
            precision = self.get_precision(field_name)
            recall = self.get_recall(field_name)
            return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            # Average F1 across all fields
            f1_scores = [self.get_f1(field) for field in self.field_names]
            return statistics.mean(f1_scores) if f1_scores else 0.0
    
    def get_accuracy(self, field_name: str = None) -> float:
        """
        Calculate accuracy for a field or average across all fields.
        
        Args:
            field_name: Optional field name to calculate accuracy for
            
        Returns:
            float: Accuracy value between 0 and 1
        """
        if field_name:
            if self.total[field_name] == 0:
                return 0.0
            return self.true_positives[field_name] / self.total[field_name]
        else:
            # Average accuracy across all fields
            accuracies = [self.get_accuracy(field) for field in self.field_names]
            return statistics.mean(accuracies) if accuracies else 0.0
    
    def get_metrics_by_field(self) -> Dict[str, Dict[str, float]]:
        """
        Get all metrics organized by field.
        
        Returns:
            Dict: Field metrics with precision, recall, F1, and accuracy for each field
        """
        metrics = {}
        for field in self.field_names:
            metrics[field] = {
                "precision": self.get_precision(field),
                "recall": self.get_recall(field),
                "f1": self.get_f1(field),
                "accuracy": self.get_accuracy(field)
            }
        return metrics


class DocumentMetrics:
    """
    Calculate and store document-level metrics.
    """
    
    def __init__(self):
        """Initialize document metrics calculator."""
        self.reset()
    
    def reset(self):
        """Reset all metrics counters."""
        self.correct_documents = 0
        self.total_documents = 0
    
    def update(self, correct: bool):
        """
        Update metrics with a new document result.
        
        Args:
            correct: Whether all fields in the document were correctly extracted
        """
        self.total_documents += 1
        if correct:
            self.correct_documents += 1
    
    def get_accuracy(self) -> float:
        """
        Calculate the overall document-level accuracy.
        
        Returns:
            float: Accuracy value between 0 and 1
        """
        return self.correct_documents / self.total_documents if self.total_documents > 0 else 0.0


class SchemaValidationMetrics:
    """
    Calculate and store schema validation metrics.
    """
    
    def __init__(self):
        """Initialize schema validation metrics calculator."""
        self.reset()
    
    def reset(self):
        """Reset all metrics counters."""
        self.valid_schemas = 0
        self.invalid_schemas = 0
    
    def update(self, valid: bool):
        """
        Update metrics with a new schema validation result.
        
        Args:
            valid: Whether the schema validation was successful
        """
        if valid:
            self.valid_schemas += 1
        else:
            self.invalid_schemas += 1
    
    def get_success_rate(self) -> float:
        """
        Calculate the schema validation success rate.
        
        Returns:
            float: Success rate between 0 and 1
        """
        total = self.valid_schemas + self.invalid_schemas
        return self.valid_schemas / total if total > 0 else 0.0


class ExecutionTimeMetrics:
    """
    Calculate and store execution time metrics.
    """
    
    def __init__(self):
        """Initialize execution time metrics calculator."""
        self.reset()
    
    def reset(self):
        """Reset all metrics data."""
        self.times = []
    
    def update(self, time_seconds: float):
        """
        Update metrics with a new execution time.
        
        Args:
            time_seconds: Execution time in seconds
        """
        self.times.append(time_seconds)
    
    def get_average_time(self) -> float:
        """
        Calculate the average execution time.
        
        Returns:
            float: Average time in seconds
        """
        return statistics.mean(self.times) if self.times else 0.0
    
    def get_min_time(self) -> float:
        """
        Get the minimum execution time.
        
        Returns:
            float: Minimum time in seconds
        """
        return min(self.times) if self.times else 0.0
    
    def get_max_time(self) -> float:
        """
        Get the maximum execution time.
        
        Returns:
            float: Maximum time in seconds
        """
        return max(self.times) if self.times else 0.0 
import time
from typing import Type, Dict, Any, List, Optional, Union
import json
from datetime import datetime

from pydantic import BaseModel
import deepeval
from deepeval.metrics import JsonCorrectnessMetric, ExactMatchMetric, F1Metric
from deepeval.test_case import LLMTestCase

from extract_thinker import Extractor, Contract
from extract_thinker.eval.dataset import EvaluationDataset
from extract_thinker.eval.metrics import (
    FieldMetrics,
    DocumentMetrics,
    SchemaValidationMetrics,
    ExecutionTimeMetrics
)
from extract_thinker.eval.report import EvaluationReport


class Evaluator:
    """
    The main class for evaluating ExtractThinker's extraction capabilities.
    
    This evaluator runs a set of test cases against a dataset and produces
    a comprehensive report of the extraction performance.
    """
    
    def __init__(
        self,
        extractor: Extractor,
        response_model: Type[Contract],
        vision: bool = False,
        content: Optional[str] = None
    ):
        """
        Initialize the evaluator.
        
        Args:
            extractor: An initialized Extractor instance to use for extraction
            response_model: The Contract class that defines the expected output schema
            vision: Whether to use vision mode for extraction
            content: Optional extra content to prepend to the extraction input
        """
        self.extractor = extractor
        self.response_model = response_model
        self.vision = vision
        self.content = content
        
        # Initialize metrics
        self.field_metrics = FieldMetrics(response_model)
        self.document_metrics = DocumentMetrics()
        self.schema_metrics = SchemaValidationMetrics()
        self.time_metrics = ExecutionTimeMetrics()
        
        # Store test results
        self.results = []
        self.evaluation_name = None
        self.dataset_name = None
        
    def evaluate(
        self,
        dataset: EvaluationDataset,
        evaluation_name: str = "Extraction Evaluation",
        skip_failures: bool = False
    ) -> EvaluationReport:
        """
        Evaluate extraction on the provided dataset.
        
        Args:
            dataset: Dataset containing documents and expected outputs
            evaluation_name: Name of this evaluation run
            skip_failures: Whether to continue after schema validation failures
            
        Returns:
            EvaluationReport: A complete report of the evaluation results
        """
        self.evaluation_name = evaluation_name
        self.dataset_name = dataset.name
        self.results = []
        
        # Reset metrics
        self.field_metrics.reset()
        self.document_metrics.reset()
        self.schema_metrics.reset()
        self.time_metrics.reset()
        
        # Process each document in the dataset
        for doc_id, doc_path, expected in dataset.items():
            # Extract data from document
            start_time = time.time()
            try:
                result = self.extractor.extract(
                    source=doc_path,
                    response_model=self.response_model,
                    vision=self.vision,
                    content=self.content
                )
                schema_valid = True
                predicted = result.dict()
            except Exception as e:
                schema_valid = False
                predicted = {"error": str(e)}
                
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Update schema validation metrics
            self.schema_metrics.update(schema_valid)
            self.time_metrics.update(execution_time)
            
            # If schema is invalid and we're not skipping failures, continue to next doc
            if not schema_valid and not skip_failures:
                self.results.append({
                    "doc_id": doc_id,
                    "expected": expected,
                    "predicted": predicted,
                    "fields_correct": {},
                    "schema_valid": schema_valid,
                    "execution_time_s": execution_time
                })
                continue
                
            # Compute field-level metrics
            fields_correct = {}
            for field_name in expected.keys():
                if field_name in predicted:
                    # Check if values match
                    field_correct = self._values_match(expected[field_name], predicted[field_name])
                    fields_correct[field_name] = field_correct
                    # Update metrics
                    self.field_metrics.update(field_name, field_correct, True)
                else:
                    # Field is missing in predicted output
                    fields_correct[field_name] = False
                    self.field_metrics.update(field_name, False, False)
            
            # Check if all fields are correct (document-level accuracy)
            doc_correct = all(fields_correct.values())
            self.document_metrics.update(doc_correct)
            
            # Store result for this document
            self.results.append({
                "doc_id": doc_id,
                "expected": expected,
                "predicted": predicted,
                "fields_correct": fields_correct,
                "schema_valid": schema_valid,
                "execution_time_s": execution_time
            })
            
        # Generate and return the evaluation report
        return self._generate_report()
    
    def _values_match(self, expected: Any, predicted: Any) -> bool:
        """
        Check if the expected and predicted values match.
        
        This implements exact matching for simple values and more complex
        matching for lists and dictionaries.
        
        Args:
            expected: The expected value from ground truth
            predicted: The predicted value from extraction
            
        Returns:
            bool: True if values match, False otherwise
        """
        # Handle None values
        if expected is None and predicted is None:
            return True
        
        # Handle lists
        if isinstance(expected, list) and isinstance(predicted, list):
            # If lengths differ, not an exact match
            if len(expected) != len(predicted):
                return False
            
            # Check each item
            for exp_item, pred_item in zip(expected, predicted):
                if not self._values_match(exp_item, pred_item):
                    return False
            return True
        
        # Handle dictionaries
        if isinstance(expected, dict) and isinstance(predicted, dict):
            # If keys differ, not an exact match
            if set(expected.keys()) != set(predicted.keys()):
                return False
            
            # Check each key-value pair
            for key in expected:
                if not self._values_match(expected[key], predicted[key]):
                    return False
            return True
        
        # For simple values, use exact string comparison after converting to strings
        return str(expected).strip() == str(predicted).strip()
    
    def _generate_report(self) -> EvaluationReport:
        """
        Generate a complete evaluation report.
        
        Returns:
            EvaluationReport: The evaluation report with all metrics
        """
        # Get model name from the extractor
        model_name = "unknown"
        if hasattr(self.extractor, "llm") and hasattr(self.extractor.llm, "model"):
            model_name = self.extractor.llm.model
        
        report = EvaluationReport(
            evaluation_name=self.evaluation_name,
            dataset=self.dataset_name or "Custom Dataset",
            model=model_name,
            timestamp=datetime.now().isoformat(),
            metrics={
                "documents_tested": len(self.results),
                "overall_document_accuracy": self.document_metrics.get_accuracy(),
                "schema_validation_rate": self.schema_metrics.get_success_rate(),
                "average_precision": self.field_metrics.get_precision(),
                "average_recall": self.field_metrics.get_recall(),
                "average_f1": self.field_metrics.get_f1(),
                "average_execution_time_s": self.time_metrics.get_average_time()
            },
            field_metrics=self.field_metrics.get_metrics_by_field(),
            results=self.results
        )
        
        return report
    
    def save_report(self, report: EvaluationReport, output_path: str) -> None:
        """
        Save the evaluation report to a JSON file.
        
        Args:
            report: The evaluation report to save
            output_path: Path to save the JSON report
        """
        with open(output_path, 'w') as f:
            f.write(report.model_dump_json(indent=2))
        
        print(f"Evaluation report saved to {output_path}") 
import time
from typing import Type, Dict, Any, Optional, Union, Callable
from datetime import datetime
from extract_thinker import Extractor, Contract
from extract_thinker.eval.dataset import EvaluationDataset
from extract_thinker.eval.metrics import (
    FieldMetrics,
    DocumentMetrics,
    SchemaValidationMetrics,
    ExecutionTimeMetrics
)
from extract_thinker.eval.report import EvaluationReport
from extract_thinker.eval.field_comparison import (
    FieldComparisonManager,
    FieldComparisonConfig,
    ComparisonType
)
from extract_thinker.eval.hallucination import HallucinationDetector
from extract_thinker.eval.cost_metrics import CostMetrics
from litellm import completion_cost, token_counter
from extract_thinker.eval.hallucination import HallucinationDetector
from extract_thinker.eval.HallucinationDetectionStrategy import HallucinationDetectionStrategy
import uuid


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
        content: Optional[str] = None,
        field_comparisons: Optional[Dict[str, Union[ComparisonType, FieldComparisonConfig]]] = None,
        detect_hallucinations: bool = False,
        track_costs: bool = False,
        document_text_provider: Optional[Callable[[str], str]] = None
    ):
        """
        Initialize the evaluator.
        
        Args:
            extractor: An initialized Extractor instance to use for extraction
            response_model: The Contract class that defines the expected output schema
            vision: Whether to use vision mode for extraction
            content: Optional extra content to prepend to the extraction input
            field_comparisons: Optional dict mapping field names to comparison types or configs
            detect_hallucinations: Whether to detect potential hallucinations
            track_costs: Whether to track token usage and costs
            document_text_provider: Optional function that takes a document path and returns its text content
        """
        self.extractor = extractor
        self.response_model = response_model
        self.vision = vision
        self.content = content
        
        # Initialize comparison manager
        self.field_comparison_manager = FieldComparisonManager(response_model)
        
        # Configure field comparisons if provided
        if field_comparisons:
            for field_name, comparison in field_comparisons.items():
                if isinstance(comparison, ComparisonType):
                    self.field_comparison_manager.set_comparison(
                        field_name, 
                        FieldComparisonConfig(comparison_type=comparison)
                    )
                elif isinstance(comparison, FieldComparisonConfig):
                    self.field_comparison_manager.set_comparison(field_name, comparison)
        
        # Initialize metrics
        self.field_metrics = FieldMetrics(response_model)
        self.document_metrics = DocumentMetrics()
        self.schema_metrics = SchemaValidationMetrics()
        self.time_metrics = ExecutionTimeMetrics()
        
        # Store test results
        self.results = []
        self.evaluation_name = None
        self.dataset_name = None
        
        # Initialize hallucination detection if enabled
        self.detect_hallucinations = detect_hallucinations
        self.hallucination_detector = None
        if detect_hallucinations:
            # Get LLM from extractor if available
            llm = getattr(extractor, "llm", None)
            
            # Create detector with appropriate strategy
            if llm:
                strategy = HallucinationDetectionStrategy.LLM
                print(f"Using LLM strategy for hallucination detection with {llm.model if hasattr(llm, 'model') else 'unknown model'}")
            else:
                strategy = HallucinationDetectionStrategy.HEURISTIC
                print("Using heuristic strategy for hallucination detection (no LLM available)")
                
            try:
                self.hallucination_detector = HallucinationDetector(
                    llm=llm,
                    strategy=strategy
                )
                print("Hallucination detector initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize hallucination detector: {str(e)}")
        
        # Initialize cost tracking if enabled
        self.track_costs = track_costs
        self.cost_metrics = CostMetrics()
        
        # Store document text provider to allow custom loading of document content
        self.document_text_provider = document_text_provider
        
    def set_field_comparison(
        self, 
        field_name: str, 
        comparison_type: ComparisonType,
        similarity_threshold: float = 0.8,
        numeric_tolerance: float = 0.01,
        custom_comparator: Optional[Callable[[Any, Any], bool]] = None
    ):
        """
        Set comparison configuration for a specific field.
        
        Args:
            field_name: Name of the field
            comparison_type: Type of comparison to use
            similarity_threshold: Threshold for similarity-based comparisons
            numeric_tolerance: Tolerance for numeric comparisons
            custom_comparator: Custom comparison function for CUSTOM comparison type
        """
        config = FieldComparisonConfig(
            comparison_type=comparison_type,
            similarity_threshold=similarity_threshold,
            numeric_tolerance=numeric_tolerance,
            custom_comparator=custom_comparator
        )
        self.field_comparison_manager.set_comparison(field_name, config)
        
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
            result = self._extract_document(doc_id, doc_path, expected, skip_failures)
            if result:
                self.results.append(result)
        
        # Generate model name
        model_name = "unknown"
        if hasattr(self.extractor, "llm") and hasattr(self.extractor.llm, "model"):
            model_name = self.extractor.llm.model
            
        # Create a report
        metrics = {
            "documents_tested": len(self.results),
            "overall_document_accuracy": self.document_metrics.get_accuracy(),
            "schema_validation_rate": self.schema_metrics.get_success_rate(),
            "average_precision": self.field_metrics.get_precision(),
            "average_recall": self.field_metrics.get_recall(),
            "average_f1": self.field_metrics.get_f1(),
            "average_execution_time_s": self.time_metrics.get_average_time()
        }
        
        # Add cost metrics if tracked
        if self.track_costs:
            cost_metrics = self.cost_metrics.get_metrics()
            metrics.update(cost_metrics)
        
        report = EvaluationReport(
            evaluation_name=self.evaluation_name,
            dataset=self.dataset_name or "Custom Dataset",
            model=model_name,
            timestamp=datetime.now().isoformat(),
            documents_evaluated=len(self.results),
            metrics=metrics,
            field_metrics=self.field_metrics.get_metrics_by_field(),
            results=self.results,
            comparison_configs={
                field: config.dict() 
                for field, config in self.field_comparison_manager.field_configs.items()
            }
        )
        
        return report
    
    def _extract_document(self, doc_id: str, doc_path: str, expected: Dict[str, Any], skip_failures: bool) -> Optional[Dict[str, Any]]:
        """
        Extract data from a single document and evaluate against expected output.
        
        Args:
            doc_id: Identifier for the document
            doc_path: Path to the document
            expected: Expected output
            skip_failures: Whether to continue after schema validation failures
            
        Returns:
            Dict or None: Result dictionary or None if extraction failed and skip_failures is False
        """
        print(f"Processing document: {doc_id}")
        
        # Get document text for hallucination detection if needed using provided document_text_provider
        document_text = None
        if self.detect_hallucinations:
            if self.document_text_provider:
                try:
                    document_text = self.document_text_provider(doc_path)
                except Exception as e:
                    print(f"Warning: Could not extract text from {doc_id} using document_text_provider: {str(e)}")
            elif hasattr(self.extractor, 'document_loader') and self.extractor.document_loader:
                try:
                    # Try to use the extractor's document loader as fallback
                    pages = self.extractor.document_loader.load(doc_path)
                    document_text = "\n".join(page.get("content", "") for page in pages)
                except Exception as e:
                    print(f"Warning: Could not extract text using extractor's document_loader: {str(e)}")
                    
            if document_text is None:
                print(f"Warning: No document text available for hallucination detection for {doc_id}")
        
        # Time the extraction
        start_time = time.time()
        input_tokens = 0
        output_tokens = 0
        cost = 0.0
        
        try:
            # Track input size if cost tracking is enabled
            if self.track_costs and hasattr(self.extractor, "llm") and hasattr(self.extractor.llm, "model"):
                # Estimate input tokens
                try:
                    if document_text:
                        input_tokens = token_counter(model=self.extractor.llm.model, text=document_text)
                except:
                    pass
            
            # Extract data from document
            extracted = self.extractor.extract(
                doc_path, 
                self.response_model, 
                vision=self.vision,
                content=self.content
            )
            
            schema_valid = True
            self.schema_metrics.update(True)
            
            # Convert to dict for comparison
            if hasattr(extracted, "dict"):
                try:
                    predicted = extracted.dict()
                except AttributeError:
                    # For Pydantic v2
                    predicted = extracted.model_dump()
            else:
                predicted = dict(extracted)
            
            # Track token usage and cost
            if self.track_costs and hasattr(self.extractor, "llm") and hasattr(extracted, "_response"):
                try:
                    # Get token usage from response
                    usage = extracted._response.usage
                    input_tokens = usage.prompt_tokens
                    output_tokens = usage.completion_tokens
                    
                    # Calculate cost
                    cost = completion_cost(
                        completion_response=extracted._response,
                        model=self.extractor.llm.model
                    )
                    
                    # Update cost metrics
                    self.cost_metrics.update(doc_id, input_tokens, output_tokens, cost)
                except Exception as e:
                    print(f"Warning: Could not track cost for {doc_id}: {str(e)}")
                
        except Exception as e:
            print(f"Extraction failed for {doc_id}: {str(e)}")
            schema_valid = False
            self.schema_metrics.update(False)
            
            if not skip_failures:
                raise
                
            # Return failure result
            return {
                "doc_id": doc_id,
                "expected": expected,
                "predicted": None,
                "fields_correct": {},
                "schema_valid": False,
                "execution_time_s": time.time() - start_time,
                "error": str(e),
                "tokens": {"input": input_tokens, "output": output_tokens, "total": input_tokens + output_tokens},
                "cost": cost
            }
                
        # Record execution time
        execution_time = time.time() - start_time
        self.time_metrics.update(execution_time)
                
        # Compute field-level metrics using field-specific comparison methods
        fields_correct = {}
        for field_name in expected.keys():
            if field_name in predicted:
                # Check if values match using appropriate comparison
                expected_value = expected[field_name]
                predicted_value = predicted[field_name]
                
                field_correct = self.field_comparison_manager.compare_values(
                    field_name, expected_value, predicted_value
                )
                
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
        
        # Detect hallucinations if enabled
        hallucination_results = None
        if self.detect_hallucinations and self.hallucination_detector and document_text:
            try:
                hallucination_results = self.hallucination_detector.detect_hallucinations(
                    predicted, document_text
                )
                print(f"Hallucination detection completed for {doc_id}")
            except Exception as e:
                print(f"Warning: Hallucination detection failed for {doc_id}: {str(e)}")
        elif self.detect_hallucinations:
            if not self.hallucination_detector:
                print(f"Warning: Hallucination detection enabled but detector not initialized for {doc_id}")
            if document_text is None:
                print(f"Warning: Hallucination detection enabled but no document text available for {doc_id}")
        
        # Return result for this document
        result = {
            "doc_id": doc_id,
            "expected": expected,
            "predicted": predicted,
            "fields_correct": fields_correct,
            "schema_valid": schema_valid,
            "execution_time_s": execution_time,
            "comparison_results": {
                field: {
                    "expected": expected.get(field),
                    "predicted": predicted.get(field) if field in predicted else None,
                    "comparison_type": self.field_comparison_manager.get_comparison(field).comparison_type.value,
                    "matched": fields_correct.get(field, False)
                } 
                for field in expected.keys()
            }
        }
        
        # Add hallucination results if available
        if hallucination_results:
            try:
                # Try both dict methods to support both Pydantic v1 and v2
                try:
                    result["hallucination_results"] = hallucination_results.dict()
                except AttributeError:
                    # For Pydantic v2
                    result["hallucination_results"] = hallucination_results.model_dump()
            except Exception as e:
                print(f"Warning: Could not add hallucination results to output for {doc_id}: {str(e)}")
                # Fallback to a simple representation
                result["hallucination_results"] = {
                    "overall_score": hallucination_results.overall_score,
                    "field_scores": hallucination_results.field_scores
                }
        
        # Add token usage and cost if tracked
        if self.track_costs:
            result["tokens"] = {"input": input_tokens, "output": output_tokens, "total": input_tokens + output_tokens}
            result["cost"] = cost
            
        return result
    
    def save_report(self, report: EvaluationReport, output_path: str):
        """
        Save the evaluation report to a file.
        
        Args:
            report: Evaluation report to save
            output_path: Path to save the report
        """
        with open(output_path, 'w') as f:
            f.write(report.json(indent=2))
        print(f"Report saved to: {output_path}")

class TeacherStudentEvaluator(Evaluator):
    """
    An evaluator that compares a standard extractor (student) against a superior extractor (teacher).
    
    This allows benchmarking extractor performance against a more capable model 
    to identify potential areas for improvement.
    """
    
    def __init__(
        self,
        student_extractor: Extractor,
        teacher_extractor: Extractor,
        response_model: Type[Contract],
        student_vision: bool = False,
        teacher_vision: bool = False,
        student_content: Optional[str] = None,
        teacher_content: Optional[str] = None
    ):
        """
        Initialize the teacher-student evaluator.
        
        Args:
            student_extractor: The standard extractor to evaluate
            teacher_extractor: The superior extractor to compare against
            response_model: The Contract class that defines the expected output schema
            student_vision: Whether to use vision mode for the student extractor
            teacher_vision: Whether to use vision mode for the teacher extractor
            student_content: Optional extra content for the student extractor
            teacher_content: Optional extra content for the teacher extractor
        """
        # Initialize the base evaluator with the student extractor
        super().__init__(
            extractor=student_extractor,
            response_model=response_model,
            vision=student_vision,
            content=student_content
        )
        
        # Store teacher extractor settings
        self.teacher_extractor = teacher_extractor
        self.teacher_vision = teacher_vision
        self.teacher_content = teacher_content
        
        # Initialize additional metrics for teacher and comparison
        self.teacher_field_metrics = FieldMetrics(response_model)
        self.teacher_document_metrics = DocumentMetrics()
        self.teacher_schema_metrics = SchemaValidationMetrics()
        self.teacher_time_metrics = ExecutionTimeMetrics()
        
        # Store teacher results
        self.teacher_results = []
    
    def _values_match(self, expected: Any, predicted: Any) -> bool:
        """
        Helper method to check if values match using the field comparison manager.
        This uses a simple equality comparison as a fallback when the field name is not known.
        
        Args:
            expected: Expected value
            predicted: Predicted value
            
        Returns:
            bool: True if values match, False otherwise
        """
        # Use simple equality as a fallback comparison method
        if isinstance(expected, (str, int, float, bool)) and isinstance(predicted, (str, int, float, bool)):
            return expected == predicted
        elif isinstance(expected, dict) and isinstance(predicted, dict):
            # For dictionaries, check if all keys match
            return all(self._values_match(expected.get(k), predicted.get(k)) for k in set(expected.keys()) | set(predicted.keys()))
        elif isinstance(expected, list) and isinstance(predicted, list):
            # For lists, check if lengths match and all items match
            if len(expected) != len(predicted):
                return False
            return all(self._values_match(e, p) for e, p in zip(expected, predicted))
        else:
            # For other types, use string representation equality
            return str(expected) == str(predicted)
    
    def evaluate(
        self,
        dataset: EvaluationDataset,
        evaluation_name: str = "Teacher-Student Evaluation",
        skip_failures: bool = False
    ) -> EvaluationReport:
        """
        Evaluate both student and teacher extractors on the provided dataset.
        
        Args:
            dataset: Dataset containing documents and expected outputs
            evaluation_name: Name of this evaluation run
            skip_failures: Whether to continue after schema validation failures
            
        Returns:
            EvaluationReport: A complete report with comparative metrics
        """
        self.evaluation_name = evaluation_name
        self.dataset_name = dataset.name
        self.results = []
        self.teacher_results = []
        
        # Reset all metrics
        self.field_metrics.reset()
        self.document_metrics.reset()
        self.schema_metrics.reset()
        self.time_metrics.reset()
        
        self.teacher_field_metrics.reset()
        self.teacher_document_metrics.reset()
        self.teacher_schema_metrics.reset()
        self.teacher_time_metrics.reset()
        
        # Process each document in the dataset
        for doc_id, doc_path, expected in dataset.items():
            # Student extraction
            student_result = self._extract_with_extractor(
                self.extractor, 
                doc_path, 
                expected, 
                self.vision, 
                self.content,
                self.field_metrics,
                self.document_metrics,
                self.schema_metrics,
                self.time_metrics,
                skip_failures
            )
            self.results.append(student_result)
            
            # Teacher extraction
            teacher_result = self._extract_with_extractor(
                self.teacher_extractor,
                doc_path,
                expected,
                self.teacher_vision,
                self.teacher_content,
                self.teacher_field_metrics,
                self.teacher_document_metrics,
                self.teacher_schema_metrics,
                self.teacher_time_metrics,
                skip_failures
            )
            self.teacher_results.append(teacher_result)
            
        # Generate and return the comparative report
        return self._generate_comparative_report()
    
    def _extract_with_extractor(
        self,
        extractor: Extractor,
        doc_path: str,
        expected: Dict[str, Any],
        vision: bool,
        content: Optional[str],
        field_metrics: FieldMetrics,
        document_metrics: DocumentMetrics,
        schema_metrics: SchemaValidationMetrics,
        time_metrics: ExecutionTimeMetrics,
        skip_failures: bool,
        doc_id: str = str(uuid.uuid4())
    ) -> Dict[str, Any]:
        """Helper method to extract data using a specific extractor and update metrics."""
        # Get document text for hallucination detection if needed using provided document_text_provider
        document_text = None
        if self.detect_hallucinations and self.document_text_provider:
            try:
                document_text = self.document_text_provider(doc_path)
            except Exception as e:
                print(f"Warning: Could not extract text from {doc_id} using document_text_provider: {str(e)}")
        
        # Extract data from document
        start_time = time.time()
        
        # Initialize token tracking variables if needed
        input_tokens = 0
        output_tokens = 0
        cost = 0.0
        
        try:
            # Track input size if cost tracking is enabled
            if self.track_costs and hasattr(extractor, "llm") and hasattr(extractor.llm, "model"):
                # Estimate input tokens
                try:
                    if document_text:
                        input_tokens = token_counter(model=extractor.llm.model, text=document_text)
                except:
                    pass
            
            # Extract data from document
            result = extractor.extract(
                source=doc_path,
                response_model=self.response_model,
                vision=vision,
                content=content
            )
            
            schema_valid = True
            predicted = result.dict()
            
            # Track token usage and cost
            if self.track_costs and hasattr(extractor, "llm") and hasattr(result, "_response"):
                try:
                    # Get token usage from response
                    usage = result._response.usage
                    input_tokens = usage.prompt_tokens
                    output_tokens = usage.completion_tokens
                    
                    # Calculate cost
                    cost = completion_cost(
                        completion_response=result._response,
                        model=extractor.llm.model
                    )
                    
                    # Update cost metrics
                    self.cost_metrics.update(doc_id, input_tokens, output_tokens, cost)
                except Exception as e:
                    print(f"Warning: Could not track cost for {doc_id}: {str(e)}")
                
        except Exception as e:
            print(f"Extraction failed for {doc_id}: {str(e)}")
            schema_valid = False
            predicted = {"error": str(e)}
                
        end_time = time.time()
        execution_time = end_time - start_time
            
        # Update schema validation metrics
        schema_metrics.update(schema_valid)
        time_metrics.update(execution_time)
            
        # If schema is invalid and we're not skipping failures, return early result
        if not schema_valid and not skip_failures:
            return {
                "doc_id": doc_id,
                "expected": expected,
                "predicted": predicted,
                "fields_correct": {},
                "schema_valid": schema_valid,
                "execution_time_s": execution_time
            }
                
        # Compute field-level metrics
        fields_correct = {}
        for field_name in expected.keys():
            if field_name in predicted:
                # Check if values match
                field_correct = self._values_match(expected[field_name], predicted[field_name])
                fields_correct[field_name] = field_correct
                # Update metrics
                field_metrics.update(field_name, field_correct, True)
            else:
                # Field is missing in predicted output
                fields_correct[field_name] = False
                field_metrics.update(field_name, False, False)
        
        # Check if all fields are correct (document-level accuracy)
        doc_correct = all(fields_correct.values())
        document_metrics.update(doc_correct)
        
        # Detect hallucinations if enabled
        hallucination_results = None
        if self.detect_hallucinations and self.hallucination_detector and document_text:
            hallucination_results = self.hallucination_detector.detect_hallucinations(
                predicted, document_text
            )
        
        # Return result for this document
        result = {
            "doc_id": doc_id,
            "expected": expected,
            "predicted": predicted,
            "fields_correct": fields_correct,
            "schema_valid": schema_valid,
            "execution_time_s": execution_time
        }
        
        # Add hallucination results if available
        if hallucination_results:
            result["hallucination_results"] = hallucination_results.dict()
        
        # Add token usage and cost if tracked
        if self.track_costs:
            result["tokens"] = {"input": input_tokens, "output": output_tokens, "total": input_tokens + output_tokens}
            result["cost"] = cost
            
        return result
    
    def _generate_comparative_report(self) -> EvaluationReport:
        """
        Generate a comprehensive evaluation report with teacher-student comparison.
        
        Returns:
            EvaluationReport: The evaluation report with comparative metrics
        """
        # Get model names
        student_model = "unknown"
        teacher_model = "unknown"
        
        if hasattr(self.extractor, "llm") and hasattr(self.extractor.llm, "model"):
            student_model = self.extractor.llm.model
            
        if hasattr(self.teacher_extractor, "llm") and hasattr(self.teacher_extractor.llm, "model"):
            teacher_model = self.teacher_extractor.llm.model
        
        # Create comparative metrics
        field_improvements = {}
        for field_name in self.field_metrics.field_names:
            student_f1 = self.field_metrics.get_f1(field_name)
            teacher_f1 = self.teacher_field_metrics.get_f1(field_name)
            
            # Calculate improvement percentage
            if student_f1 > 0:
                improvement = ((teacher_f1 - student_f1) / student_f1) * 100
            else:
                improvement = float('inf') if teacher_f1 > 0 else 0
                
            field_improvements[field_name] = {
                "student_f1": student_f1,
                "teacher_f1": teacher_f1,
                "improvement_pct": improvement
            }
        
        # Calculate overall improvement
        student_doc_accuracy = self.document_metrics.get_accuracy()
        teacher_doc_accuracy = self.teacher_document_metrics.get_accuracy()
        
        if student_doc_accuracy > 0:
            doc_accuracy_improvement = ((teacher_doc_accuracy - student_doc_accuracy) / student_doc_accuracy) * 100
        else:
            doc_accuracy_improvement = float('inf') if teacher_doc_accuracy > 0 else 0
        
        # Create the report with comparative metrics
        report = EvaluationReport(
            evaluation_name=self.evaluation_name,
            dataset=self.dataset_name or "Custom Dataset",
            model=f"Student: {student_model}, Teacher: {teacher_model}",
            timestamp=datetime.now().isoformat(),
            documents_evaluated=len(self.results),
            metrics={
                "student_document_accuracy": student_doc_accuracy,
                "teacher_document_accuracy": teacher_doc_accuracy,
                "document_accuracy_improvement": doc_accuracy_improvement,
                "student_schema_validation_rate": self.schema_metrics.get_success_rate(),
                "teacher_schema_validation_rate": self.teacher_schema_metrics.get_success_rate(),
                "student_average_execution_time": self.time_metrics.get_average_time(),
                "teacher_average_execution_time": self.teacher_time_metrics.get_average_time()
            },
            field_improvements=field_improvements,
            field_metrics=self.field_metrics.get_metrics_by_field(),
            teacher_field_metrics=self.teacher_field_metrics.get_metrics_by_field(),
            results=self.results,
            teacher_results=self.teacher_results
        )
        
        return report 
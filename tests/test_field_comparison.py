import os
import pytest
from extract_thinker import Extractor, Contract
from extract_thinker.eval import (
    Evaluator, 
    FileSystemDataset, 
    ComparisonType,
    FieldComparisonConfig
)
from extract_thinker.document_loader.document_loader_pypdf import DocumentLoaderPyPdf
from extract_thinker.global_models import get_lite_model
from typing import List

# Define a test contract
class InvoiceContract(Contract):
    invoice_number: str  # Needs exact matching
    description: str     # Can use semantic similarity
    total_amount: float  # Can use numeric tolerance
    notes: str           # Can use fuzzy matching

def test_field_comparison_types():
    # Set up paths
    cwd = os.getcwd()
    test_file_path = os.path.join(cwd, "tests", "files", "invoice.pdf")
    
    # Create mock dataset
    dataset = MockDataset(
        documents={
            "invoice.pdf": {
                "path": test_file_path,
                "expected": {
                    "invoice_number": "00012",
                    "description": "Professional consulting services for Q1 2023",
                    "total_amount": 1125.0,
                    "notes": "Thank you for your business!"
                }
            }
        }
    )
    
    # Initialize extractor
    extractor = Extractor()
    extractor.load_document_loader(DocumentLoaderPyPdf())
    extractor.load_llm(get_lite_model())
    
    # Set up evaluator with different field comparison types
    evaluator = Evaluator(
        extractor=extractor,
        response_model=InvoiceContract,
        field_comparisons={
            "invoice_number": ComparisonType.EXACT,
            "description": ComparisonType.SEMANTIC,
            "total_amount": ComparisonType.NUMERIC,
            "notes": ComparisonType.FUZZY
        }
    )
    
    # Configure specific thresholds
    evaluator.set_field_comparison(
        "description", 
        ComparisonType.SEMANTIC,
        similarity_threshold=0.7  # Lower threshold for more lenient matching
    )
    
    evaluator.set_field_comparison(
        "total_amount", 
        ComparisonType.NUMERIC,
        numeric_tolerance=0.05  # 5% tolerance
    )
    
    # Run evaluation
    report = evaluator.evaluate(dataset)
    
    # Print summary
    report.print_summary()
    
    # Verify results are as expected
    assert "invoice_number" in report.field_metrics
    assert "description" in report.field_metrics
    assert "total_amount" in report.field_metrics
    assert "notes" in report.field_metrics
    
    # Check that comparison types are correctly reported
    assert report.comparison_configs["invoice_number"]["comparison_type"] == "exact"
    assert report.comparison_configs["description"]["comparison_type"] == "semantic"
    assert report.comparison_configs["total_amount"]["comparison_type"] == "numeric"
    assert report.comparison_configs["notes"]["comparison_type"] == "fuzzy"

# Mock dataset for testing
class MockDataset(FileSystemDataset):
    def __init__(self, documents):
        self.name = "Mock Dataset"
        self.documents = documents
    
    def items(self):
        for doc_id, data in self.documents.items():
            yield doc_id, data["path"], data["expected"] 
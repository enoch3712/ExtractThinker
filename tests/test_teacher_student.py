import unittest
from unittest.mock import MagicMock, patch
import tempfile
import os
import json
from typing import List, Dict, Any

from extract_thinker import Extractor, Contract
from extract_thinker.eval import TeacherStudentEvaluator, FileSystemDataset


class TestContract(Contract):
    invoice_number: str
    date: str
    total_amount: float
    line_items: List[Dict[str, Any]] = []


class TestTeacherStudentEvaluator(unittest.TestCase):
    def setUp(self):
        # Create mock extractors
        self.student_extractor = MagicMock(spec=Extractor)
        self.student_extractor.llm = MagicMock()
        self.student_extractor.llm.model = "gpt-4o-mini"
        
        self.teacher_extractor = MagicMock(spec=Extractor)
        self.teacher_extractor.llm = MagicMock()
        self.teacher_extractor.llm.model = "gpt-4o"
        
        # Create a temp directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a simple dataset
        self.labels = {
            "doc1.pdf": {
                "invoice_number": "INV-001",
                "date": "2023-01-15",
                "total_amount": 100.50,
                "line_items": [{"item": "Service A", "amount": 100.50}]
            }
        }
        
        # Create labels.json in temp dir
        with open(os.path.join(self.temp_dir.name, "labels.json"), "w") as f:
            json.dump(self.labels, f)
        
        # Create a dummy PDF file
        with open(os.path.join(self.temp_dir.name, "doc1.pdf"), "w") as f:
            f.write("dummy pdf content")
        
        # Set up dataset
        self.dataset = FileSystemDataset(
            documents_dir=self.temp_dir.name,
            labels_path=os.path.join(self.temp_dir.name, "labels.json"),
            name="Test Dataset"
        )
        
        # Mock student extraction result - less accurate
        self.student_result = TestContract(
            invoice_number="INV-002",  # Incorrect
            date="2023-01-15", 
            total_amount=100.00,  # Slightly off
            line_items=[{"item": "Service A", "amount": 100.00}]
        )
        
        # Mock teacher extraction result - more accurate
        self.teacher_result = TestContract(
            invoice_number="INV-001",
            date="2023-01-15",
            total_amount=100.50,
            line_items=[{"item": "Service A", "amount": 100.50}]
        )
        
        # Configure extractors to return our mock results
        self.student_extractor.extract.return_value = self.student_result
        self.teacher_extractor.extract.return_value = self.teacher_result
        
    def tearDown(self):
        self.temp_dir.cleanup()
        
    @patch("extract_thinker.eval.cost_metrics.CostMetrics.track_cost")
    def test_teacher_student_evaluation(self, mock_track_cost):
        """Test teacher-student comparative evaluation"""
        # Mock cost tracking results
        mock_track_cost.side_effect = [
            # Student cost
            {
                "prompt_tokens": 300,
                "completion_tokens": 100,
                "total_tokens": 400,
                "cost_usd": 0.0080
            },
            # Teacher cost
            {
                "prompt_tokens": 500,
                "completion_tokens": 200, 
                "total_tokens": 700,
                "cost_usd": 0.0175
            }
        ]
        
        evaluator = TeacherStudentEvaluator(
            student_extractor=self.student_extractor,
            teacher_extractor=self.teacher_extractor,
            response_model=TestContract,
            track_costs=True
        )
        
        report = evaluator.evaluate(self.dataset)
        
        # Verify both extractors were called
        self.student_extractor.extract.assert_called_once()
        self.teacher_extractor.extract.assert_called_once()
        
        # Check document metrics
        self.assertEqual(report.documents_evaluated, 1)
        
        # Check model metrics
        self.assertIn("student_document_accuracy", report.metrics)
        self.assertIn("teacher_document_accuracy", report.metrics)
        
        # Model-specific accuracy (student should be lower)
        self.assertLess(report.metrics["student_document_accuracy"], 
                        report.metrics["teacher_document_accuracy"])
        
        # Check improvement metrics
        self.assertIn("document_accuracy_improvement", report.metrics)
        self.assertGreater(report.metrics["document_accuracy_improvement"], 0)
        
        # Check cost metrics
        self.assertIn("student_average_cost", report.metrics)
        self.assertIn("teacher_average_cost", report.metrics)
        self.assertEqual(report.metrics["student_average_cost"], 0.0080)
        self.assertEqual(report.metrics["teacher_average_cost"], 0.0175)
        
        # Check field-level improvements
        self.assertIn("invoice_number", report.field_improvements)
        self.assertIn("total_amount", report.field_improvements)
        
        # Invoice number should show improvement (student got it wrong)
        self.assertGreater(report.field_improvements["invoice_number"]["improvement_pct"], 0) 
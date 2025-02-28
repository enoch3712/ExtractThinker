import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os
import json
import sys

from extract_thinker.eval.cli import load_contract, main


class SimpleContract:
    """Dummy contract for testing"""
    pass


class TestCLI(unittest.TestCase):
    def setUp(self):
        # Create a temp directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a simple contract file
        contract_code = """
from extract_thinker import Contract
from typing import List, Dict, Any

class TestContract(Contract):
    invoice_number: str
    date: str
    total_amount: float
    line_items: List[Dict[str, Any]] = []
"""
        self.contract_path = os.path.join(self.temp_dir.name, "test_contract.py")
        with open(self.contract_path, "w") as f:
            f.write(contract_code)
            
        # Create a simple config file
        config = {
            "evaluation_name": "Test Evaluation",
            "dataset_name": "Test Dataset",
            "contract_path": self.contract_path,
            "documents_dir": self.temp_dir.name,
            "labels_path": os.path.join(self.temp_dir.name, "labels.json"),
            "file_pattern": "*.pdf",
            "llm": {
                "model": "gpt-4o-mini",
                "params": {
                    "temperature": 0.0
                }
            },
            "detect_hallucinations": True,
            "track_costs": True
        }
        
        self.config_path = os.path.join(self.temp_dir.name, "config.json")
        with open(self.config_path, "w") as f:
            json.dump(config, f)
            
        # Create a simple labels file
        labels = {
            "test.pdf": {
                "invoice_number": "INV-001",
                "date": "2023-01-15",
                "total_amount": 100.50,
                "line_items": []
            }
        }
        
        with open(os.path.join(self.temp_dir.name, "labels.json"), "w") as f:
            json.dump(labels, f)
            
        # Create a dummy PDF file
        with open(os.path.join(self.temp_dir.name, "test.pdf"), "w") as f:
            f.write("dummy pdf content")
            
    def tearDown(self):
        self.temp_dir.cleanup()
        
    @patch("importlib.util.spec_from_file_location")
    @patch("importlib.util.module_from_spec")
    def test_load_contract(self, mock_module_from_spec, mock_spec_from_file):
        """Test dynamic contract loading"""
        # Set up the mocks
        mock_spec = MagicMock()
        mock_spec_from_file.return_value = mock_spec
        
        mock_module = MagicMock()
        mock_module.TestContract = SimpleContract
        mock_module_from_spec.return_value = mock_module
        
        # Call the function
        contract_class = load_contract(self.contract_path)
        
        # Verify the expected calls
        mock_spec_from_file.assert_called_once()
        mock_module_from_spec.assert_called_once()
        mock_spec.loader.exec_module.assert_called_once()
        
        # Check the result
        self.assertEqual(contract_class, SimpleContract)
        
    @patch("extract_thinker.eval.cli.load_contract")
    @patch("extract_thinker.eval.cli.Extractor")
    @patch("extract_thinker.eval.cli.Evaluator")
    @patch("extract_thinker.eval.cli.FileSystemDataset")
    @patch("sys.argv", ["extract_thinker-eval", "--config", "config.json", "--output", "results.json"])
    def test_main_cli(self, mock_dataset, mock_evaluator, mock_extractor, mock_load_contract):
        """Test the CLI main function"""
        # Set up mocks
        mock_load_contract.return_value = SimpleContract
        
        mock_extractor_instance = MagicMock()
        mock_extractor.return_value = mock_extractor_instance
        
        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance
        
        mock_evaluator_instance = MagicMock()
        mock_evaluator_instance.evaluate.return_value = MagicMock()
        mock_evaluator.return_value = mock_evaluator_instance
        
        # Call main function
        with patch("os.path.exists", return_value=True):
            with patch("json.load", return_value=json.loads(open(self.config_path).read())):
                with patch("builtins.open", create=True):
                    main()
        
        # Verify the calls
        mock_load_contract.assert_called_once()
        mock_extractor.assert_called_once()
        mock_dataset.assert_called_once()
        mock_evaluator.assert_called_once()
        mock_evaluator_instance.evaluate.assert_called_once()
        mock_evaluator_instance.save_report.assert_called_once() 
import unittest
from unittest.mock import MagicMock, patch

from extract_thinker.eval.hallucination import HallucinationDetector


class TestHallucinationDetector(unittest.TestCase):
    def setUp(self):
        # Create mock LLM
        self.mock_llm = MagicMock()
        
        # Sample document text and extracted data
        self.document_text = """
        INVOICE
        Invoice Number: INV-001
        Date: January 15, 2023
        Customer: ACME Corp
        
        Item: Consulting Services
        Amount: $100.50
        
        Total: $100.50
        """
        
        self.extracted_data = {
            "invoice_number": "INV-001",
            "date": "2023-01-15",
            "customer": "ACME Corp",
            "total_amount": 100.50,
            "fake_field": "This is hallucinated"  # Field not in document
        }
        
    def test_hallucination_detection(self):
        """Test that hallucination detector correctly identifies hallucinations"""
        # Mock LLM response for hallucination checking
        llm_response = {
            "invoice_number": {
                "hallucination_score": 0.1,
                "reasoning": "The invoice number INV-001 is clearly present in the document."
            },
            "date": {
                "hallucination_score": 0.2,
                "reasoning": "The date January 15, 2023 is in the document, though formatted differently."
            },
            "customer": {
                "hallucination_score": 0.1,
                "reasoning": "ACME Corp is clearly mentioned as the customer."
            },
            "total_amount": {
                "hallucination_score": 0.1,
                "reasoning": "The total amount $100.50 is present in the document."
            },
            "fake_field": {
                "hallucination_score": 0.9,
                "reasoning": "There is no mention of 'This is hallucinated' anywhere in the document."
            }
        }
        
        # Configure mock LLM to return our prepared response
        self.mock_llm.generate.return_value = {"choices": [{"message": {"content": json.dumps(llm_response)}}]}
        
        # Create detector
        detector = HallucinationDetector(llm=self.mock_llm)
        
        # Run detection
        with patch('json.loads', return_value=llm_response):
            results = detector.detect_hallucinations(
                extracted_data=self.extracted_data, 
                document_text=self.document_text
            )
        
        # Verify LLM was called
        self.mock_llm.generate.assert_called_once()
        
        # Check overall score (should be influenced by the hallucinated field)
        self.assertGreater(results["overall_score"], 0.2)
        
        # Check field scores
        self.assertEqual(results["field_scores"]["invoice_number"], 0.1)
        self.assertEqual(results["field_scores"]["fake_field"], 0.9)
        
        # Test with custom threshold
        detector = HallucinationDetector(llm=self.mock_llm, threshold=0.8)
        self.assertEqual(detector.threshold, 0.8) 
import os
import sys
import unittest
from datetime import datetime
import pandas as pd

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from examples.receipt_processor import (
    ReceiptItem,
    ReceiptContract,
    setup_extractor,
    process_receipt
)

class TestReceiptProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.extractor = setup_extractor()
        cls.test_receipt_path = os.path.join("examples", "files", "receipts", "receipt1.jpg")
        cls.test_receipts_dir = os.path.join("examples", "files", "receipts")
        cls.test_output_dir = os.path.join("tests", "test_output")
        
        # Create output directory if it doesn't exist
        os.makedirs(cls.test_output_dir, exist_ok=True)

    def test_receipt_item_contract(self):
        """Test ReceiptItem contract structure"""
        item = ReceiptItem(
            description="Test Item",
            quantity=2.0,
            unit_price=10.0,
            total=20.0
        )
        self.assertEqual(item.description, "Test Item")
        self.assertEqual(item.quantity, 2.0)
        self.assertEqual(item.unit_price, 10.0)
        self.assertEqual(item.total, 20.0)

    def test_receipt_contract(self):
        """Test ReceiptContract structure"""
        items = [
            ReceiptItem(description="Item 1", quantity=1, unit_price=10.0, total=10.0),
            ReceiptItem(description="Item 2", quantity=2, unit_price=15.0, total=30.0)
        ]
        receipt = ReceiptContract(
            merchant_name="Test Store",
            date="01/01/2024",
            total_amount=40.0,
            tax_amount=3.2,
            payment_method="Credit Card",
            items=items
        )
        self.assertEqual(receipt.merchant_name, "Test Store")
        self.assertEqual(receipt.date, "01/01/2024")
        self.assertEqual(receipt.total_amount, 40.0)
        self.assertEqual(receipt.tax_amount, 3.2)
        self.assertEqual(receipt.payment_method, "Credit Card")
        self.assertEqual(len(receipt.items), 2)

    def test_process_receipt(self):
        """Test processing a single receipt"""
        if os.path.exists(self.test_receipt_path):
            # Just verify that the function runs without errors
            try:
                process_receipt(self.test_receipt_path)
                self.assertTrue(True)  # If we get here, the test passed
            except Exception as e:
                self.fail(f"process_receipt raised an exception: {str(e)}")

if __name__ == '__main__':
    unittest.main() 
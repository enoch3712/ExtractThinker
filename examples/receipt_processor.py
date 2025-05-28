import os
import json
import yaml
from typing import List, Optional
from pydantic import Field

from extract_thinker import Extractor, Contract, DocumentLoaderTesseract

def json_to_yaml(json_dict):
    # Check if json_dict is a dictionary
    if not isinstance(json_dict, dict):
        raise ValueError("json_dict must be a dictionary")

    # Convert the Python dictionary to YAML
    yaml_str = yaml.dump(json_dict)
    return yaml_str

class ReceiptItem(Contract):
    """Represents a single item in a receipt"""
    description: str = Field("Description of the item purchased")
    quantity: float = Field("Quantity of items purchased")
    unit_price: float = Field("Price per unit")
    total: float = Field("Total price for this item (quantity * unit_price)")

class ReceiptContract(Contract):
    """Defines the structure of data to extract from receipts"""
    merchant_name: str = Field("Name of the store or business")
    date: str = Field("Date of purchase in format DD/MM/YYYY")
    total_amount: float = Field("Total amount of the receipt including tax")
    tax_amount: Optional[float] = Field("Tax amount if specified on receipt")
    payment_method: Optional[str] = Field("Method of payment (e.g., cash, credit card, etc.)")
    items: List[ReceiptItem] = Field("List of items purchased")

def setup_extractor():
    """Initialize and configure the extractor"""
    extractor = Extractor()
    
    # Use Tesseract for OCR
    tesseract_path = "/opt/homebrew/bin/tesseract"  # Update this path for your system
    extractor.load_document_loader(DocumentLoaderTesseract(tesseract_path))
    
    # Configure LLM
    extractor.load_llm("gpt-4")
    
    return extractor

def process_receipt(receipt_path: str):
    """Process a receipt and display its information"""
    extractor = setup_extractor()
    
    print(f"\nProcessing receipt: {os.path.basename(receipt_path)}")
    print("-" * 50)
    
    result = extractor.extract(receipt_path, ReceiptContract)
    
    # Convert to JSON and validate
    try:
        receipt_json = json.loads(result.model_dump_json())
        if not isinstance(receipt_json, dict):
            raise ValueError("Extracted data is not a valid dictionary")
        
        # Convert to YAML for better readability
        yaml_output = json_to_yaml(receipt_json)
        print("\nExtracted Receipt Data (YAML format):")
        print("-" * 50)
        print(yaml_output)
        
        return receipt_json
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error processing receipt: {str(e)}")
        return None

def main():
    # Process the receipt
    receipt_path = os.path.join("examples", "files", "receipts", "receipt1.jpg")
    if os.path.exists(receipt_path):
        receipt_data = process_receipt(receipt_path)
        if receipt_data:
            print("\nReceipt processed successfully!")
    else:
        print(f"Receipt not found at {receipt_path}")

if __name__ == "__main__":
    main() 
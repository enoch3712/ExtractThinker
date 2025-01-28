import os
from typing import List
from dotenv import load_dotenv
from extract_thinker import Extractor
from extract_thinker.document_loader.document_loader_pypdf import DocumentLoaderPyPdf
from extract_thinker.models.contract import Contract
from pydantic import BaseModel, field_validator

class InvoiceLine(BaseModel):
    description: str
    quantity: int
    unit_price: float
    amount: float

    @field_validator('quantity', mode='before')
    def convert_quantity_to_int(cls, v):
        if isinstance(v, float):
            return int(v)
        return v

class InvoiceContract(Contract):
    invoice_number: str
    invoice_date: str
    lines: List[InvoiceLine]
    total_amount: float

class CreditNoteContract(Contract):
    credit_note_number: str
    credit_note_date: str
    lines: List[InvoiceLine]
    total_amount: int

class FinancialContract(Contract):
    total_amount: int
    document_number: str
    document_date: str

load_dotenv()
cwd = os.getcwd()

def test_critical_extract_with_pypdf():
    """Critical test for basic extraction functionality"""
    test_file_path = f"{cwd}/tests/files/invoice.pdf"

    extractor = Extractor()
    extractor.load_document_loader(DocumentLoaderPyPdf())
    extractor.load_llm("groq/llama-3.3-70b-versatile")

    result = extractor.extract(test_file_path, InvoiceContract)

    assert result is not None
    assert result.lines[0].description == "Consultation services"
    assert result.lines[0].quantity == 3
    assert result.lines[0].unit_price == 375
    assert result.lines[0].amount == 1125
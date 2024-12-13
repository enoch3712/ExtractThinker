from typing import List
from extract_thinker.models.contract import Contract
from pydantic import BaseModel, Field, field_validator

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
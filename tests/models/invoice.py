from typing import List
from extract_thinker.models.contract import Contract

class LinesContract(Contract):
    description: str
    quantity: int
    unit_price: int
    amount: int

class InvoiceContract(Contract):
    invoice_number: str
    invoice_date: str
    lines: List[LinesContract]
    total_amount: int

class CreditNoteContract(Contract):
    credit_note_number: str
    credit_note_date: str
    lines: List[LinesContract]
    total_amount: int

class FinancialContract(Contract):
    total_amount: int
    document_number: str
    document_date: str
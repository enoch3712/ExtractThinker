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

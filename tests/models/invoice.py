from extract_thinker.models.contract import Contract


class InvoiceContract(Contract):
    invoice_number: str
    invoice_date: str

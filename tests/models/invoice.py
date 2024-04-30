from extract_thinker.models import Contract


class InvoiceContract(Contract):
    invoice_number: str
    invoice_date: str

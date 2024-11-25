# Contracts

!!! warning "🚧 In Development"
    This component is currently under active development. The API might change in future releases.

Contracts in ExtractThinker are Pydantic models that define the structure of data you want to extract. They provide type safety and validation for your extracted data.

## Basic Usage

```python
from extract_thinker import Contract
from typing import List, Optional
from pydantic import Field

class InvoiceLineItem(Contract):
    description: str = Field(description="Description of the item")
    quantity: int = Field(description="Quantity of items")
    unit_price: float = Field(description="Price per unit")
    amount: float = Field(description="Total amount for line")

class InvoiceContract(Contract):
    invoice_number: str = Field(description="Invoice identifier")
    date: str = Field(description="Invoice date")
    total_amount: float = Field(description="Total invoice amount")
    line_items: List[InvoiceLineItem] = Field(description="List of items in invoice")
    notes: Optional[str] = Field(description="Additional notes", default=None)
```

??? example "Base Contract Implementation"
    ```python
    --8<-- "extract_thinker/models/contract.py"
    ```

For more examples and advanced usage, check out the [examples directory](examples/) in the repository.

# Contracts

!!! warning "🚧 In Development"
    This component is currently under active development. The API might change in future releases.

Contracts in ExtractThinker are Pydantic models that define the structure of data you want to extract. They provide type safety and validation for your extracted data.

??? example "Base Contract Implementation"
    ```python
    --8<-- "extract_thinker/models/contract.py"
    ```

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
## Enrich a contract after extraction

Use Pydantic's `model_validator(mode="after")` to derive application-owned fields
once the extracted fields have been parsed. This can map a tax identifier to an
internal record, for example:

```python
from typing import Optional
from pydantic import model_validator
from extract_thinker import Contract

# Replace this example mapping with your application's lookup.
customer_ids = {"ACME-TAX-ID": 42}

class InvoiceContract(Contract):
    tax_id: str
    internal_id: Optional[int] = None

    @model_validator(mode="after")
    def attach_internal_id(self):
        self.internal_id = customer_ids.get(self.tax_id)
        return self
```

Pass this contract to `extractor.extract(...)` as usual. The returned object
contains the enrichment. Validators may run more than once during validation
or retries: keep them idempotent and avoid writes or other irreversible side
effects. For expensive or asynchronous database work, perform the lookup after
`extract()` returns and assign the result explicitly in your application.

## Bounding boxes and signatures

`BoundingBox`, `DocumentRegion` and `Signature` are reusable Pydantic types:

```python
from extract_thinker import Contract, Signature

class SignedAgreement(Contract):
    signature: Signature
```

`Signature` contains `present`, optional `signer`, optional `bounding_box` and
optional `confidence` (0–1). It describes a signature mark; it does not verify
a person's identity or validate a digital signature.

`BoundingBox` contains a one-based `page` and normalized `x0`, `y0`, `x1`, `y1`
coordinates from the top-left of the rendered page. Inverted or out-of-range
rectangles fail validation. `DocumentRegion` pairs source text with a bounding
box and optional provider confidence.

The [PyMuPDF loader](../document-loaders/pymupdf.md) can supply actual text-region
coordinates with `include_bbox=True`. Typed fields alone do not make every
loader capable of supplying coordinates, and any coordinates inferred by an
LLM should be checked against the source.

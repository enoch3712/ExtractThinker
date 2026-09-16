# Local entity masking

`EntityMasker` replaces configured entity values and regex matches with opaque placeholders locally. Use `DocumentLoaderMasked` to mask loaded text before it reaches the extraction model, then restore placeholders in the result.

```python
from extract_thinker import (
    Contract, Extractor, DocumentLoaderPyMuPDF, DocumentLoaderMasked, EntityMasker,
)

class Contact(Contract):
    name: str
    email: str

masker = EntityMasker(
    entities={"PERSON": ["Alice Smith", "Bob Jones"]},
    patterns={"CUSTOMER_ID": r"CUST-\d+"},
    detect_emails=True,
)
loader = DocumentLoaderMasked(DocumentLoaderPyMuPDF(), masker)
extractor = Extractor(loader)
extractor.load_llm("your-provider/your-model")
masked_result = extractor.extract("contacts.pdf", Contact)
result = Contact.model_validate(loader.restore(masked_result.model_dump()))
```

Exact supplied values are matched case-sensitively at word boundaries, preferring the longest match at a shared starting position. Patterns use Python regex syntax. Email matching is a convenience recognizer; names, addresses, phone numbers and other sensitive data require explicit values or patterns. This is not a complete PII detector or an anonymization guarantee.

## Scope and restoration

Masking walks string values in lists, tuples and dictionaries, including text, tables, forms and region text. Dictionary keys and non-string values are unchanged. Inspect the loaded representation and configure rules for the data you need to protect. The wrapper runs after document loading, so a cloud OCR loader may already have sent the original document to its own service.

A wrapper keeps one session across load calls, allowing repeated values in multiple pages/files to use the same placeholder. Create a wrapper per extraction job or call `loader.reset()` when finished. Resetting discards the restoration mapping. The wrapper restores only exact placeholders; altered or invented placeholders remain unchanged.

For direct use:

```python
masked = masker.mask("Email Alice Smith at alice@example.com")
print(masked.content)
original = masked.restore(masked.content)
```

`MaskedContent.mapping` contains original values and should be kept local. Its repr omits the mapping. Every independent `mask()` call gets a fresh namespace; use `masker.session()` when several inputs need shared identities.

Use string fields for masked entities. A placeholder is not a valid email address, phone number or other format-constrained value; restore the plain data before validating those formats with your final application model. Masking changes model input and may affect extraction quality.

## Images

The text wrapper rejects vision mode and removes `image`/`images` page fields, including cached images. It does not redact image pixels or PDF files. Use a separately redacted image workflow if you need visual extraction of private documents.

The APIs are available on main after release 0.1.14. No masking calls are sent to an LLM.

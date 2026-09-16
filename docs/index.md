---
hide:
  - toc
---

<div class="et-hero" markdown>

<span class="et-eyebrow">OPEN SOURCE DOCUMENT INTELLIGENCE</span>

# Complex documents.<br>Validated Python objects.

ExtractThinker turns documents into typed data using your choice of parser and LLM. Define the result with Pydantic, then load, classify, split and extract.

[Run your first extraction](getting-started/index.md){ .md-button .md-button--primary }
[Run the MCP container](getting-started/mcp-container.md){ .md-button }

</div>

```python
class Invoice(Contract):
    invoice_number: str
    total: float

result = extractor.extract("invoice.pdf", Invoice)
```

A contract defines the output shape. Pydantic validates the result; your application decides how to handle missing data, conflicts and business rules.

## Build the workflow your documents need

<div class="grid cards" markdown>

- **Read with the right parser**

    Text PDFs, scanned pages, tables, spreadsheets and cloud OCR. Keep page metadata alongside content.

    [Choose a loader](core-concepts/document-loaders/index.md)

- **Handle documents that span pages**

    Select source pages, retrieve relevant text or extract pages independently and validate their combined result.

    [Understand completion strategies](core-concepts/completion-strategies/index.md)

- **Use different models for different work**

    Route by document complexity or assign fields to independent extraction groups with their own model and vision settings.

    [Extract fields in parallel](core-concepts/extractors/parallel-fields.md)

- **Keep application logic in Python**

    Add contract validators, local entity masking, page events and callbacks. Expose extraction through MCP when you need a service.

    [Explore the 2026 update](getting-started/2026-update.md)

</div>

## Start small. Verify with your own documents.

The quickstart includes a credential-free loader check and a small invoice example. Model accuracy, latency and cost depend on your documents and provider. Schema validation is a useful gate; it is not a factual-accuracy guarantee.

[Quickstart](getting-started/index.md) · [Local Ollama setup](core-concepts/llm-integration/ollama.md) · [Contribute on GitHub](https://github.com/enoch3712/ExtractThinker/blob/main/CONTRIBUTING.md)

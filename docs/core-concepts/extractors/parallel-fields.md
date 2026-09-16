# Partial and parallel field extraction

Use `FieldExtraction` annotations to give top-level contract fields different models, vision settings or instructions. `Extractor.extract()` recognizes these annotations automatically, extracts the fields in separate calls, and validates the assembled contract.

```python
from typing import Annotated
from extract_thinker import Contract, Extractor, LLM, FieldExtraction

transcription = LLM("your-provider/transcription-model", token_limit=6000)
analysis = LLM("your-provider/reasoning-model", token_limit=2000)

class Report(Contract):
    transcript: Annotated[str, FieldExtraction(
        model=transcription,
        instructions="Transcribe the source text accurately, preserving Markdown.",
    )]
    total: Annotated[float, FieldExtraction(model=analysis)]
    chart_summary: Annotated[str, FieldExtraction(model=analysis, vision=True)]

extractor = Extractor(document_loader=loader, llm=analysis)
report = extractor.extract("report.pdf", Report)
```

A string `model` creates a new `LLM` with default settings. Pass a configured `LLM` instance to retain custom output limits, provider options or backend settings. A field without a model override uses the extractor's LLM. `vision=None` inherits the call's vision setting; `True` or `False` overrides it for that group. Images are loaded if any group needs them and are only included in vision requests.

## Group related fields

Fields with the same nonempty `group` are extracted together. Every field in that group must use an identical policy. Group coupled values when their relationship is easier for the model to understand in one request:

```python
totals = FieldExtraction(group="totals", model=analysis)

class Invoice(Contract):
    subtotal: Annotated[float, totals]
    tax: Annotated[float, totals]
    total: Annotated[float, totals]
    notes: Annotated[str, FieldExtraction(model=transcription)]
```

Each ungrouped field is a separate call, including unannotated fields when the contract opts into this strategy. Annotate the outer field to extract a nested object as one unit; nested field policies are not independently scheduled.

## Explicit parallel extraction

Use this method for an ordinary contract without annotations, or to control concurrency:

```python
result = extractor.extract_fields(
    "document.pdf",
    MyContract,
    max_workers=3,
    vision=False,
)
```

`max_workers` limits concurrent field groups (default 4). The document is loaded once per source before calls start, and each worker receives independent extractor/LLM state. Underlying clients and user-provided interceptors may be shared; use thread-safe clients/callbacks, or `max_workers=1` for stateful integrations. A shared `ComplexityRouter` serializes its routed calls.

Both `content` and `completion_strategy` are supported. Pagination/continuation operate inside each field group and may add calls; their own page concurrency is separate from `max_workers`. Parallel extraction repeats source context across groups, so it can increase token usage. Group fields or use page retrieval when that is preferable.

## Validation and failures

Partial schemas preserve aliases, descriptions and constraints. Top-level custom validators and defaults/factories are deferred to final assembly. Cross-field validation receives the complete result. A failed field/group raises an error naming the group; no partial contract is returned. Already-running provider requests may finish even after another group fails.

The parent's `last_completion` is cleared because several worker completions cannot be represented as one response. These APIs are available on main after release 0.1.14.

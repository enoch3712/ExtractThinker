# Route requests by complexity

`ComplexityRouter` chooses a configured model for each request using measurable input and contract features. It works as an `LLM` with `Extractor` and the splitters. It does not discover provider capabilities or predict accuracy/cost; configure models you have evaluated for your workload.

```python
from extract_thinker import ComplexityRouter, ModelRoute, LLM, Extractor

router = ComplexityRouter([
    ModelRoute(LLM("your-provider/small-model", token_limit=2000), max_score=4),
    ModelRoute(LLM("your-provider/large-vision-model", token_limit=8000), supports_vision=True),
])
extractor = Extractor(document_loader=loader, llm=router)
result = extractor.extract("document.pdf", MyContract)
print(router.last_decision)
```

Routes are evaluated in order. The first route whose threshold covers the score and whose declared capabilities support the input is selected. Thresholds must be nonnegative and strictly increasing. An optional unbounded route (`max_score=None`) must be last. Image requests skip routes unless `supports_vision=True`; declare that flag only for an actually capable model. If no route fits, the router raises before calling a provider.

## Default score

The default heuristic is:

```text
text characters / 4000
+ image count × 4
+ schema field count / 4
+ schema object depth
+ max(0, page count - 1) / 2
```

Image data URLs are counted as images, not thousands of text characters. Contract metrics include nested model references, with recursive schemas bounded to avoid loops. This score is not a token estimate or a guarantee of semantic difficulty. Thresholds require calibration against your own documents and model evaluations.

Supply a custom scorer to apply your own policy:

```python
router = ComplexityRouter(
    routes,
    scorer=lambda request: request.field_count + request.image_count * 10,
)
```

The scorer receives `RequestComplexity` with `text_characters`, `image_count`, `field_count`, `schema_depth`, and `page_count`. It must return a finite nonnegative number. `last_decision` records the selected model, score and features; `last_completion` forwards available response metadata from the selected LLM.

## Operational behavior

Provider failures propagate. Selection does not automatically retry an expensive model after a cheaper model fails. If needed, configure a LiteLLM fallback router on a route's individual `LLM`.

The router supports structured `request` and `raw_completion`. Raw calls have no response model to inspect, so schema metrics are zero; text and image metrics still apply. Temperature, thinking, dynamic mode and timeout setters are forwarded to every route. Configure provider-specific options and output limits on the route's `LLM` instance.

A single router serializes its routed calls. Use separate router instances for independent concurrent workers. Provider batch processing is rejected because it would bypass per-request routing. These APIs are available on main after release 0.1.14.

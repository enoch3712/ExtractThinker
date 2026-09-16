# Document events and rules

Wrap a loader with `DocumentLoaderEvents` to detect page signals and call application handlers. The built-in `VisionEventDetector` uses a configured vision model to identify handwriting, charts and embedded photos/illustrations. Custom rules can also run locally without a detector.

```python
from extract_thinker import (
    DocumentLoaderPyMuPDF, DocumentLoaderEvents, VisionEventDetector,
    EventRule, LLM,
)

def record_event(event, page):
    print(event.type.value, event.page_number, event.rule_name)

loader = DocumentLoaderEvents(
    DocumentLoaderPyMuPDF(),
    detector=VisionEventDetector(LLM("your-provider/vision-model")),
    on_event=record_event,
    rules=[EventRule(
        name="needs_handwriting_review",
        predicate=lambda page, signals: signals.contains_handwriting,
    )],
)
pages = loader.load("application.pdf")
print(loader.last_events)
```

The detector requires page images. The wrapper temporarily enables rendering on the underlying loader and restores its previous vision setting afterward. It makes one structured model request per page, including all images for that page. The model must actually support vision; visual detection quality depends on that model and should be evaluated on your documents. These calls may send page images to the configured provider and incur usage charges.

## Events and callbacks

Built-in event types are `handwriting`, `chart`, and `image`. A page may produce more than one. Image detection means photos or illustrations in the document, not merely the fact that the page was rendered as an image.

Each `DocumentEvent` contains a one-based source `page_number`, `type`, brief `summary`, and optional `rule_name`. Results are attached to each page as an `events` list and are preserved in normal extraction metadata. `last_events` provides the most recent document's event list.

`on_event(event, page)` receives every event. A rule can additionally provide its own handler:

```python
rule = EventRule(
    "contains_account_number",
    lambda page, signals: "Account number" in page.get("content", ""),
    handler=lambda event, page: print("Review page", event.page_number),
)
loader = DocumentLoaderEvents(base_loader, rules=[rule])
```

Without a detector, custom predicates receive `signals=None`. This supports deterministic rules over text, tables or other loaded metadata without model calls. Rule names must be unique. Handlers execute synchronously in source-page order.

All detection and predicate evaluation completes before handlers run. A detector/predicate failure propagates and fires no callbacks for a partially detected document. Handler failures also propagate, but side effects from already-called handlers cannot be rolled back. Cached underlying pages do not suppress events: each wrapper `load()` evaluates rules and invokes handlers again.

## Custom detectors

Supply an object implementing `detect(page) -> PageSignals`. Set `requires_vision=True` if it needs rendered images. `PageSignals` requires `contains_handwriting`, `contains_charts`, and `contains_images`, with an optional `summary`. Use this interface for a local computer-vision model or signals from your document parser. Invalid or missing detector results raise rather than silently reporting no detections.

These APIs are available on main after release 0.1.14. Automated tests verify event routing, image requests and callback behavior with deterministic detector responses; they do not benchmark a live vision model's accuracy.

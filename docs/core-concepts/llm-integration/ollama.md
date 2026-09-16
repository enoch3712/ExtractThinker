# Large documents with Ollama

A model returning a summary instead of your fields is a contract failure. ExtractThinker raises a validation error rather than treating that output as a successful extraction. Three separate settings matter: input context, output budget, and structured output support.

## Use native structured output

`structured_output=True` asks Instructor to send the Pydantic contract as a native JSON Schema response format. Use the `ollama_chat/` provider for Ollama's chat endpoint:

```python
from pydantic import Field
from extract_thinker import Contract, DocumentLoaderDocling, Extractor, LLM

class Notification(Contract):
    notification_number: int = Field(description="Numero del oficio de notificacion")

llm = LLM(
    "ollama_chat/qwen2.5vl:latest",  # Use a model installed on your server.
    structured_output=True,
    token_limit=1000,
    completion_kwargs={
        "api_base": "http://localhost:11434",
        "num_ctx": 32768,
    },
)
extractor = Extractor(DocumentLoaderDocling(), llm)
result = extractor.extract(
    ["page-1.jpg", "page-2.jpg", "page-3.jpg"],
    Notification,
    vision=False,
)
print(result.notification_number)
```

With `vision=False`, the loader must produce usable text/OCR. With `vision=True`, use a loader that supplies page images and an installed vision-capable model. Check the loaded content before debugging the LLM.

Native schema support depends on the provider, model and server version. It improves output shape; it does not guarantee that extracted values are correct. The default remains Markdown JSON mode for compatibility. Native mode is available on the default LiteLLM backend and cannot be combined with dynamic parsing or the raw LiteLLM router; configure direct LLM instances as routes in [ComplexityRouter](complexity-router.md) instead. Raw completion methods do not enforce a contract.

See Ollama's [structured output documentation](https://docs.ollama.com/capabilities/structured-outputs) for server capabilities. The library still validates the returned object even when the provider claims schema enforcement.

## Check the actual input context

`token_limit` limits the response, not the input context. Set `num_ctx` to a value supported by the installed model and available memory, then inspect `ollama ps` to confirm the allocated context. Larger contexts require more memory. See [Ollama context configuration](https://docs.ollama.com/context-length).

The report in [issue #357](https://github.com/enoch3712/ExtractThinker/issues/357) showed exactly 4,096 prompt tokens when seven pages failed. That is consistent with an input-context limit, but it does not prove truncation; the original private documents and server were unavailable for reproduction. The regression tests verify seven-page prompt preservation, schema transmission, provider options and rejection of wrong-shaped responses using an offline transport. They do not benchmark the accuracy of those local models.

## Reduce the document per call

If the full document does not fit, use [pagination](../completion-strategies/paginate.md), [page selection](../document-loaders/pypdf.md), or [retrieval](../extractors/retrieval.md). Pagination extracts pages independently and validates the merged result. Conflicting scalar values may require another model call; unresolved conflicts and failed pages raise instead of returning incomplete data.

[Concatenation](../completion-strategies/concatenate.md) handles a response that runs out of output tokens. It does not increase the input context window. A smaller document succeeding is a reason to inspect context limits and loaded content before replacing the contract or making required fields optional.

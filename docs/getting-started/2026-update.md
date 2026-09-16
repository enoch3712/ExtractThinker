# The 2026 update

These changes are available on `main`; package metadata remains `0.1.14` until a separate release is prepared. Installing the existing PyPI release does not install this update. Use the source-checkout instructions in the [quickstart](index.md).

## New capabilities

| Workflow | What is available |
| --- | --- |
| Deploy a service | [MCP and Docker](mcp-container.md): HTTP/stdio, JSON Schema contracts and bounded access to a document directory |
| Extract selected information | [SQLite page retrieval](../core-concepts/extractors/retrieval.md) and one-based page selection |
| Distribute extraction work | [Field groups and parallel extraction](../core-concepts/extractors/parallel-fields.md), per-field models and vision |
| Choose a model by workload | [ComplexityRouter](../core-concepts/llm-integration/complexity-router.md) with explicit routes and a customizable score |
| Process private text locally first | [Entity masking](../core-concepts/extractors/masking.md) with explicit entities, patterns and reversible tokens |
| React to page content | [Page events](../core-concepts/document-loaders/events.md), custom rules and an optional vision detector |
| Parse more PDF layouts | [PyMuPDF](../core-concepts/document-loaders/pymupdf.md), [Camelot/Tabula](../core-concepts/document-loaders/pdf-tables.md) and [Adobe PDF](../core-concepts/document-loaders/adobe-pdf.md) |
| Work with Markdown | [Conversion](../core-concepts/markdown-conversion/index.md) and [heading-based splitting](../core-concepts/splitters/markdown.md) |
| Enforce output shape locally | [Native JSON Schema mode and Ollama context guidance](../core-concepts/llm-integration/ollama.md) |

## Correctness and compatibility

Pagination preserves source order and field metadata, reports failed pages, and raises on unresolved conflicts. Concatenation has bounded continuation behavior. Required fields are no longer silently filled to hide incomplete extraction. Applications that relied on partial or fabricated results must handle these errors explicitly.

Numeric classification IDs distinguish classifications with the same display name. Split groups are checked for source-page coverage. Multi-file inputs load each document rather than treating the filenames as document text.

Cloud loader fixes cover AWS temporary credentials/default credential chains and Azure API versions, tables and blank cells. Docling configuration and import diagnostics have been updated. PDFium rendering supports its current API. Image prompts now use the actual image MIME type.

Python support remains **3.9–3.13** for the core package. The MCP service is optional and requires **3.10+**. Loader SDKs, native OCR dependencies and provider capabilities have their own requirements. Install only the integrations you use and consult their setup pages.

## Verification and limits

The update has 231 passing offline core tests. Targeted tests were also run on Python 3.9 and 3.13; repository installation checks cover 3.9–3.13. Real local SDK checks exercised Docling conversion, Camelot/Tabula table parsing and Adobe job construction. The optional MCP suite uses the real SDK and an offline model transport; its container was built and checked over HTTP.

Cloud adapters tested with intercepted responses are not live cloud-service benchmarks. The original private documents from the Ollama issue were unavailable. Local masking is explicit/pattern-based and does not promise complete PII detection. Retrieval ranks text lexically and still parses the source document. Vision events depend on the chosen model.

See the [resolution ledger](https://github.com/enoch3712/ExtractThinker/blob/main/planning/issue-resolution.md) for issue-by-issue evidence and the [facelift roadmap](https://github.com/enoch3712/ExtractThinker/blob/main/planning/2026-facelift.md) for work beyond the backlog.

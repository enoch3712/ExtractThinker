# Issue resolution ledger

Baseline: 29 open issues on 2026-09-16. Close only after implementation, relevant verification, and delivery to the repository. Feature requests retain their original scope. GitHub closure is a separate gate from a local passing test.

| Issue | Requirement | State | Evidence |
| --- | --- | --- | --- |
| [#357](https://github.com/enoch3712/ExtractThinker/issues/357) | Extractor is ignoring model contract with large documents | Pending | — |
| [#356](https://github.com/enoch3712/ExtractThinker/issues/356) | Optional argument "token_limit" in class LLM in llm.py not used. | Verified locally; delivery pending | Explicit completion and page-budget regression tests |
| [#352](https://github.com/enoch3712/ExtractThinker/issues/352) | How to add logprobs and top_logprobs params? | Verified locally; delivery pending | Provider option forwarding and metadata tests |
| [#351](https://github.com/enoch3712/ExtractThinker/issues/351) | Question about the Concatenate completion strategy for multi-page document | Verified locally; delivery pending | Continuation fixes and input-versus-output strategy documentation; input-size reproduction still pending |
| [#347](https://github.com/enoch3712/ExtractThinker/issues/347) | [Enhance] To support api_version param in Document Intelligence Documen Loader | Verified locally; delivery pending | Azure API-version forwarding tests |
| [#326](https://github.com/enoch3712/ExtractThinker/issues/326) | [BUG] CompletionStrategy.CONCATENATE mapping | Verified locally; delivery pending | JSON fragment, whitespace, schema-replacement and bounded retry tests |
| [#313](https://github.com/enoch3712/ExtractThinker/issues/313) | [Feature] Keep Tag in markdown (optional) | Pending | — |
| [#312](https://github.com/enoch3712/ExtractThinker/issues/312) | [Feature] RAG for Extraction (SQLite). | Pending | — |
| [#311](https://github.com/enoch3712/ExtractThinker/issues/311) | [Feature] Add images, per page, to markdown | Pending | — |
| [#310](https://github.com/enoch3712/ExtractThinker/issues/310) | [Feature] Page selection for DocumentLoader | Pending | — |
| [#309](https://github.com/enoch3712/ExtractThinker/issues/309) | No module named 'docling_core.types.doc.page' | Pending | — |
| [#287](https://github.com/enoch3712/ExtractThinker/issues/287) | [Feature] Bounding Box Capabilities | Pending | — |
| [#281](https://github.com/enoch3712/ExtractThinker/issues/281) | Add intelligent router to ExtractThinker | Pending | — |
| [#280](https://github.com/enoch3712/ExtractThinker/issues/280) | PyMuPDF DocumentLoader | Pending | — |
| [#258](https://github.com/enoch3712/ExtractThinker/issues/258) | Multiple partial calls | Pending | — |
| [#252](https://github.com/enoch3712/ExtractThinker/issues/252) | ExtractThinker MCP | Pending | — |
| [#247](https://github.com/enoch3712/ExtractThinker/issues/247) | The security token included in the request is invalid | Verified locally; delivery pending | AWS temporary credentials and credential-chain tests |
| [#235](https://github.com/enoch3712/ExtractThinker/issues/235) | Markdown Splitter Strategy | Pending | — |
| [#150](https://github.com/enoch3712/ExtractThinker/issues/150) | bad content | Verified locally; delivery pending | Azure blank-cell/multiple-table/table-only regressions |
| [#141](https://github.com/enoch3712/ExtractThinker/issues/141) | Make sure classification is right after split | Pending | — |
| [#121](https://github.com/enoch3712/ExtractThinker/issues/121) | Parallel extraction of images (and more) | Pending | — |
| [#79](https://github.com/enoch3712/ExtractThinker/issues/79) | Object types - Signatures, BoundingBoxes | Pending | — |
| [#48](https://github.com/enoch3712/ExtractThinker/issues/48) | Events: Add IDP events  | Pending | — |
| [#46](https://github.com/enoch3712/ExtractThinker/issues/46) | validator call after the llm call | Verified locally; delivery pending | Real Instructor adapter with offline transport exercises Pydantic post-validation; enrichment recipe added |
| [#37](https://github.com/enoch3712/ExtractThinker/issues/37) | Entity Masking - Mask private information | Pending | — |
| [#21](https://github.com/enoch3712/ExtractThinker/issues/21) | ExtractThinker hub - A container with a solution ready to go | Pending | — |
| [#10](https://github.com/enoch3712/ExtractThinker/issues/10) | Add Adobe PDF as a DocumentLoader | Pending | — |
| [#9](https://github.com/enoch3712/ExtractThinker/issues/9) | Add Tabula as a DocumentLoader | Pending | — |
| [#8](https://github.com/enoch3712/ExtractThinker/issues/8) | Add Camelot as a DocumentLoader | Pending | — |

## Delivery gates

- Implement each request or demonstrate existing behavior with a regression/example.
- Run offline tests and relevant integration checks; record unavailable external checks honestly.
- Update affected reference and recipe documentation.
- Commit and deliver tested changes, checking CI before merge.
- Close resolved issues with links to delivered changes; verify open-issue count at the end.

## First fix batch validation

- 67 offline tests passed on Python 3.12.9 with fresh dependency resolution, including Pydantic 2.13.5, Instructor 1.17.0, LiteLLM 1.101.0 and pypdfium2 5.13.0.
- Found and fixed undeclared NumPy import and removed PDFium document-render API during clean-install testing.
- Wheel and sdist built successfully; strict MkDocs build passed after removing placeholder navigation and linking existing EasyOCR docs.
- Cloud adapters were verified offline with SDK call interception; no paid provider calls were made.
- #351 still needs the pagination/input-budget follow-up before closure. Other untouched feature requests remain pending.

Delivery note: GitHub rejected workflow writes because the OAuth token lacks the `workflow` scope. CI edits are preserved locally and on local branch `codex/2026-ci-preparation`; source/docs fixes are published separately. Workflow authorization is requested; do not claim the new CI matrix is deployed yet.

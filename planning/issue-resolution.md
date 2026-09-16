# Issue resolution ledger

Baseline: 29 open issues on 2026-09-16. Close only after implementation, relevant verification, and delivery to the repository. Feature requests retain their original scope. GitHub closure is a separate gate from a local passing test.

| Issue | Requirement | State | Evidence |
| --- | --- | --- | --- |
| [#357](https://github.com/enoch3712/ExtractThinker/issues/357) | Extractor is ignoring model contract with large documents | Pending | — |
| [#356](https://github.com/enoch3712/ExtractThinker/issues/356) | Optional argument "token_limit" in class LLM in llm.py not used. | Closed; merged in [#364](https://github.com/enoch3712/ExtractThinker/pull/364) | Explicit completion and page-budget regression tests |
| [#352](https://github.com/enoch3712/ExtractThinker/issues/352) | How to add logprobs and top_logprobs params? | Closed; merged in [#364](https://github.com/enoch3712/ExtractThinker/pull/364) | Provider option forwarding and metadata tests |
| [#351](https://github.com/enoch3712/ExtractThinker/issues/351) | Question about the Concatenate completion strategy for multi-page document | Closed; merged in [#365](https://github.com/enoch3712/ExtractThinker/pull/365) | Seven-page vision extraction uses one page per request and merges in order; file lists loaded correctly; docs distinguish input limits from continuation |
| [#347](https://github.com/enoch3712/ExtractThinker/issues/347) | [Enhance] To support api_version param in Document Intelligence Documen Loader | Closed; merged in [#364](https://github.com/enoch3712/ExtractThinker/pull/364) | Azure API-version forwarding tests |
| [#326](https://github.com/enoch3712/ExtractThinker/issues/326) | [BUG] CompletionStrategy.CONCATENATE mapping | Closed; merged in [#364](https://github.com/enoch3712/ExtractThinker/pull/364) | JSON fragment, whitespace, schema-replacement and bounded retry tests |
| [#313](https://github.com/enoch3712/ExtractThinker/issues/313) | [Feature] Keep Tag in markdown (optional) | Closed; merged in [#366](https://github.com/enoch3712/ExtractThinker/pull/366) | Protected source tags restored verbatim; missing/reordered tags fail |
| [#312](https://github.com/enoch3712/ExtractThinker/issues/312) | [Feature] RAG for Extraction (SQLite). | Closed; merged in [#370](https://github.com/enoch3712/ExtractThinker/pull/370) | SQLite FTS5 retriever and loader wrapper; 100-page selection and full extractor prompt regression, persistence and document isolation tests |
| [#311](https://github.com/enoch3712/ExtractThinker/issues/311) | [Feature] Add images, per page, to markdown | Closed; merged in [#366](https://github.com/enoch3712/ExtractThinker/pull/366) | All source page images embedded with media types and original page labels; separate from model vision |
| [#310](https://github.com/enoch3712/ExtractThinker/issues/310) | [Feature] Page selection for DocumentLoader | Closed; merged in [#365](https://github.com/enoch3712/ExtractThinker/pull/365) | Common load_pages API; real PDF and cached-data selection tests |
| [#309](https://github.com/enoch3712/ExtractThinker/issues/309) | No module named 'docling_core.types.doc.page' | Closed; merged in [#367](https://github.com/enoch3712/ExtractThinker/pull/367) | Reproducible import diagnostics, dependency repair guide, fresh SDK conversion and page/configuration regressions |
| [#287](https://github.com/enoch3712/ExtractThinker/issues/287) | [Feature] Bounding Box Capabilities | Closed; merged in [#365](https://github.com/enoch3712/ExtractThinker/pull/365) | Source-backed normalized regions with extraction metadata propagation; PyMuPDF adapter |
| [#281](https://github.com/enoch3712/ExtractThinker/issues/281) | Add intelligent router to ExtractThinker | Closed; merged in [#371](https://github.com/enoch3712/ExtractThinker/pull/371) | Configurable complexity scoring and numeric thresholds; schema/text/page/image metrics, capability guards and routing diagnostics tested |
| [#280](https://github.com/enoch3712/ExtractThinker/issues/280) | PyMuPDF DocumentLoader | Closed; merged in [#365](https://github.com/enoch3712/ExtractThinker/pull/365) | Optional PyMuPDF loader; real text, tables, encrypted files, vision and rotation tests |
| [#258](https://github.com/enoch3712/ExtractThinker/issues/258) | Multiple partial calls | Implemented; delivery pending | FieldExtraction annotations and extract_fields; concurrent groups, per-field model/vision, aliases, pagination and final validation tests |
| [#252](https://github.com/enoch3712/ExtractThinker/issues/252) | ExtractThinker MCP | Pending | — |
| [#247](https://github.com/enoch3712/ExtractThinker/issues/247) | The security token included in the request is invalid | Closed; merged in [#364](https://github.com/enoch3712/ExtractThinker/pull/364) | AWS temporary credentials and credential-chain tests |
| [#235](https://github.com/enoch3712/ExtractThinker/issues/235) | Markdown Splitter Strategy | Closed; merged in [#366](https://github.com/enoch3712/ExtractThinker/pull/366) | MarkdownSplitter supports deterministic heading sections and semantic page grouping |
| [#150](https://github.com/enoch3712/ExtractThinker/issues/150) | bad content | Closed; merged in [#364](https://github.com/enoch3712/ExtractThinker/pull/364) | Azure blank-cell/multiple-table/table-only regressions |
| [#141](https://github.com/enoch3712/ExtractThinker/issues/141) | Make sure classification is right after split | Closed; merged in [#367](https://github.com/enoch3712/ExtractThinker/pull/367) | Numeric classification IDs propagate through text/image/Markdown splitting and Process extraction; ambiguous names and invalid groups rejected |
| [#121](https://github.com/enoch3712/ExtractThinker/issues/121) | Parallel extraction of images (and more) | Implemented; delivery pending | FieldExtraction annotations and extract_fields; concurrent groups, per-field model/vision, aliases, pagination and final validation tests |
| [#79](https://github.com/enoch3712/ExtractThinker/issues/79) | Object types - Signatures, BoundingBoxes | Closed; merged in [#365](https://github.com/enoch3712/ExtractThinker/pull/365) | Exported BoundingBox, DocumentRegion and Signature contract types; validation/round-trip tests |
| [#48](https://github.com/enoch3712/ExtractThinker/issues/48) | Events: Add IDP events  | Pending | — |
| [#46](https://github.com/enoch3712/ExtractThinker/issues/46) | validator call after the llm call | Closed; merged in [#364](https://github.com/enoch3712/ExtractThinker/pull/364) | Real Instructor adapter with offline transport exercises Pydantic post-validation; enrichment recipe added |
| [#37](https://github.com/enoch3712/ExtractThinker/issues/37) | Entity Masking - Mask private information | Pending | — |
| [#21](https://github.com/enoch3712/ExtractThinker/issues/21) | ExtractThinker hub - A container with a solution ready to go | Pending | — |
| [#10](https://github.com/enoch3712/ExtractThinker/issues/10) | Add Adobe PDF as a DocumentLoader | Closed; merged in [#369](https://github.com/enoch3712/ExtractThinker/pull/369) | Adobe SDK job integration, CSV tables, source page mapping and normalized regions; offline SDK tests |
| [#9](https://github.com/enoch3712/ExtractThinker/issues/9) | Add Tabula as a DocumentLoader | Closed; merged in [#368](https://github.com/enoch3712/ExtractThinker/pull/368) | Optional adapters; real Camelot 2.0.0 and Tabula 2.10.0 table extraction; blank pages/cells, stream cleanup, passwords and images covered |
| [#8](https://github.com/enoch3712/ExtractThinker/issues/8) | Add Camelot as a DocumentLoader | Closed; merged in [#368](https://github.com/enoch3712/ExtractThinker/pull/368) | Optional adapters; real Camelot 2.0.0 and Tabula 2.10.0 table extraction; blank pages/cells, stream cleanup, passwords and images covered |

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

## Second batch

- 104 offline tests passed on Python 3.12; strict MkDocs build passed.
- The same 104 tests also passed on Python 3.9 and 3.13; all six GitHub checks passed before merge.
- Pagination no longer drops failed pages, associates out-of-order results with the wrong source, fabricates missing required strings, or silently chooses unresolved conflicts. Partial schemas retain field metadata and aliases, with final contract validation. These improvements relate to #357; its original private Ollama/image reproduction has not been verified, so it remains open.
- GitHub confirms seven issues closed after merge of #364; 22 of the initial issues remain open.

## Third batch

- Markdown preservation/image embedding/splitting implemented with offline tests and documentation.
- Plain Markdown failures now raise instead of silently generating error comments/fallbacks; structured conversion does not mix error strings with PageContent results.
- GitHub confirms #365 merged and five more issues closed (12 total; 17 remaining before this batch).

## Fourth batch

- Docling dependency failures retain their cause, report SDK versions and provide dependency repair steps for #309.
- Fixed default OCR/table options, converter reuse, page numbering and Markdown metadata; six offline regressions added.
- Fresh Docling 2.127.0 / docling-core 2.96.1 installation passed a real, model-free HTML conversion and `uv pip check` (135 compatible packages).
- GitHub confirms #366 merged and three more issues closed (15 total; 14 remaining).
- Numeric classification IDs preserve identity through splitting and extraction, including duplicate display names. Failed calls and invalid page coverage now raise.
- 148 offline tests passed on Python 3.9, 3.12 and 3.13; strict documentation build passed.

## Fifth batch

- Camelot and Tabula PDF table loaders implemented with optional imports, table/page preservation, stream cleanup, image support, tests and documentation.
- GitHub confirms #367 merged and two more issues closed (17 total; 12 remaining).
- Real Camelot and Tabula SDK tests both passed on a generated ruled table plus blank page. Tabula used a portable Java runtime in the test environment.
- Ten new offline tests passed on Python 3.9, 3.12 and 3.13; strict docs build passed.

## Sixth batch

- Adobe PDF loader implemented with optional SDK, environment/injected credentials, normalized regions and CSV tables.
- Live Adobe service calls were not made; tests intercept the service while using real SDK job objects.
- GitHub confirms #368 merged (19 issues closed; 10 remaining).
- 166 offline tests passed on Python 3.12; eight Adobe regressions also passed on Python 3.9 and 3.13. Two real PDF Services SDK 4.3.0 compatibility checks passed using an offline service; strict docs build passed.

## Seventh batch

- SQLite retrieval selects relevant pages before model calls and preserves source metadata. The full document is still parsed; ranking is lexical.
- GitHub confirms #369 merged (20 issues closed; nine remaining).
- 179 offline tests passed on Python 3.12; 22 retrieval/page-selection checks also passed on Python 3.9 and 3.13. Strict docs build passed.

## Eighth batch

- ComplexityRouter selects user-configured LLMs using a customizable workload score and capability guards. No claim of automatic quality/cost prediction.
- GitHub confirms #370 merged (21 issues closed; eight remaining).
- 193 offline tests passed on Python 3.12; all 14 routing tests also passed on Python 3.9 and 3.13. Strict docs build passed.

## Ninth batch

- Partial/parallel extraction supports field annotations, model/vision policies, groups and final contract validation. Source loading occurs once per source.
- GitHub confirms #371 merged (22 issues closed; seven remaining).
- 203 offline tests passed on Python 3.12; all ten field-extraction tests also passed on Python 3.9 and 3.13. Strict docs build passed.

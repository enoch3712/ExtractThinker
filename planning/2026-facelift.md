# ExtractThinker 2026 facelift: from 1,597 to 10,000 stars

Prepared 2026-09-16. Proposal and repository audit, not an implemented upgrade or a growth forecast.

## Recommendation

Position ExtractThinker as **“Turn complex documents into validated Python objects.”**

Supporting promise: **“Classify, split, and extract with typed contracts, your choice of document parser, and your choice of LLM.”** Keep “an ORM for documents” as a supporting analogy. Lead with a working result rather than an architecture vocabulary or an exhaustive list of integrations.

The relaunch should combine reliable installation, a convincing document-to-object demo, refreshed documentation, and an observable maintenance cadence. Use existing parser integrations as an advantage; make reliable extraction workflows the reason to choose this library.

Interpretation: “10k” means GitHub stars. Treat stars as an outcome of adoption and contributor trust. No deadline or guarantee is implied.

## Verified baseline

| Item | Observation |
| --- | --- |
| Repository | [enoch3712/ExtractThinker](https://github.com/enoch3712/ExtractThinker) |
| Stars / forks | 1,597 / 153 at audit time |
| Open backlog | 29 issues and 5 pull requests; GitHub's aggregate count of 34 includes PRs |
| Current main | `66920c9af1b74bd20731ed7ac1cbe4794a0da21b` |
| Last main activity | August 2025; latest listed merge is #354 |
| Latest release | [v0.1.14](https://github.com/enoch3712/ExtractThinker/releases/tag/v0.1.14), published June 9, 2025 |
| PyPI | [extract-thinker](https://pypi.org/project/extract-thinker/) 0.1.14; five main commits after release tag |
| Source footprint | 67 library Python files; 226 tracked files; 59 tracked files under tests, including fixtures |
| Latest main CI | Package and docs workflows succeeded August 27, 2025; this is historical evidence, not a current compatibility test |
| Existing local checkout | `135-docx2txt-module-not-found`, 232 commits behind current main; four modified tracked files and one untracked test |
| Planning workspace | `../ExtractThinker-2026-facelift`, branch `codex/2026-facelift`, based on freshly fetched `origin/main` |

Remote references were fetched and stale references pruned. The existing checkout and its edits were preserved. The new worktree is the synchronized baseline; the original branch was not reset or merged.

## What has already improved upstream

Current main includes configurable loaders, Docling, MarkItDown, Mistral OCR, EasyOCR, Markdown conversion, model routing, reasoning-related configuration, and async wrappers. Do not scope these as entirely new features. First verify their behavior, compatibility, and documentation.

The README still emphasizes the older feature set. There is an opportunity to make existing work visible before expanding the API.

## Highest-priority findings

| Priority | Evidence | Action and completion criterion |
| --- | --- | --- |
| P0 | README installs only the base package but its first example constructs `DocumentLoaderPyPdf`; `pypdf` is absent from core dependencies and the loader explicitly requires it | Publish and test one complete installation recipe against a built wheel, with a bundled invoice and expected typed output. Start with explicit `pip install extract-thinker pypdf`; introduce a PDF extra only once implemented. |
| P0 | `.github/workflows/workflow.yml` runs two critical tests with `GROQ_API_KEY` and executes `poetry add pypdf` during CI | Make the required PR suite deterministic and runnable without secrets. Separate opt-in provider integration tests; declare test dependencies without changing the lock during CI. Verify from a fork PR. |
| P0 | [#357](https://github.com/enoch3712/ExtractThinker/issues/357), [#351](https://github.com/enoch3712/ExtractThinker/issues/351), [#326](https://github.com/enoch3712/ExtractThinker/issues/326) describe long-document schema/completion problems | Create reduced fixtures, reproduce each failure, and add regression tests before changing completion behavior. Assert schema validity and field completeness; avoid hiding missing fields through permissive defaults. |
| P1 | [#356](https://github.com/enoch3712/ExtractThinker/issues/356) concerns token limits; current `llm.py` does use `token_limit` in completion requests | Investigate context/input budgeting separately from output token limits. Do not close the report merely because the parameter appears in code. Test small-context providers and multi-page inputs. |
| P1 | Core dependencies include Playwright, python-magic/libmagic; many version ranges have no upper boundary | Measure installation/import behavior, move optional functionality behind extras where feasible, and add minimal-install plus per-extra checks. Native dependency guidance must be explicit. |
| P1 | Supported Python range is `>=3.9,<3.14`; local default interpreter is 3.14.4 | Decide the next release's support policy, then validate every advertised version in CI. Candidate: 3.10–3.14, subject to dependency and runtime tests. Do not raise the ceiling without evidence. |
| P1 | README badges still reference `Open-DocLLM`; example comments/imports drift; homepage is a JavaScript redirect | Correct names, executable examples and links; replace the redirect with a useful landing page. |
| P1 | `mkdocs.yml` contains ten `#` loader placeholders, including EasyOCR despite an implementation | Publish only supported documentation routes; move proposed integrations into a roadmap. All 48 existing Markdown navigation targets currently exist. |
| P1 | Docs workflow deploys on main only; no PR build gate | Add a PR documentation build, fix resulting warnings, and require a strict build before publishing. |
| P1 | `.pre-commit-config.yaml` matches `extractthinker` instead of `extract_thinker`; type hook references nonexistent paths and downloads an external shell script | Replace copied hooks with local, declared tooling targeting the real package. Run hooks on a representative library change. |
| P1 | Publish workflow uses an API token and can override version only inside the job | Reconcile tag, package metadata and changelog; build once, test the artifact, and move to trusted publishing when configured. Prevent arbitrary version text from being interpolated into shell code. |
| P2 | No tracked CONTRIBUTING, SECURITY, CHANGELOG, or CODE_OF_CONDUCT files were found | Add practical contributor setup, release notes, support expectations and a security-reporting route the maintainer actually monitors. |

These are source-review findings and reported issues, not claims that every report has been reproduced.

## Dependency modernization

Direct PyPI JSON checks on the audit date returned the candidates below. “Latest” is not synonymous with “compatible.” Recheck when implementing, review upstream migration notes, and update in separate batches.

| Package | Main's declared constraint | Candidate seen on PyPI | Work needed |
| --- | --- | --- | --- |
| Pydantic | `>=2.11.5` | 2.13.5 | Contract, validator, schema and serialization regressions |
| LiteLLM | `>=1.71.1` | 1.101.0 | Python >=3.10; provider parameters, routing, timeouts, token usage |
| Instructor | `>=1.8.3` | 1.17.0 | Retry, structured-output and batch compatibility |
| Pillow | `>=11.2.1,<12.0` | 12.3.0 | Major-version migration; image/OCR/rendering fixtures |
| pypdfium2 | `>=4.30.1` | 5.13.0 | Major-version migration; rendering and native wheels |
| Playwright | `>=1.52.0` | 1.63.0 | Python >=3.10; optional web-loader installation and browser setup |
| pytest | `^8.2.0` | 9.1.1 | Python >=3.10; plugins, collection and test markers |
| Ruff | pre-commit v0.1.7 | 0.16.7 | Config migration and staged rule adoption; avoid unrelated mass formatting |

Sequence: (1) support policy and reproducible baseline; (2) development/docs tooling; (3) Pydantic/Instructor/LiteLLM together with adapter tests; (4) PDF/image/native libraries; (5) optional cloud/parser extras. Review all remaining declared and locked dependencies in the implementation pass. Audit the resolved environment for advisories, record exceptions, and regenerate one canonical lock. Align or retire the separate legacy `requirements.txt` rather than maintaining inconsistent install paths.

Keep the current API working wherever possible. Use a proposed 0.2 release for intentional support-policy/API changes with migration notes; use a patch release for compatible fixes first. Version numbers are proposals until release scope is settled.

## Product and technical direction

1. **A reliable first extraction.** A five-minute quickstart with one sample invoice, one typed contract, explicit credentials, installation prerequisites and sample output. Provide a credential-free loader smoke test. Mark recorded output as recorded.
2. **Long-document correctness.** Clear behavior for truncation, pagination, concatenation, validation failure, retries and partial results. Add representative multilingual/table/mixed-document fixtures and known limitations.
3. **Evidence with extraction.** Explore optional page/source references without changing existing `Contract` return values. Bounding boxes depend on loader capability; never imply all adapters can supply them. Build on [#287](https://github.com/enoch3712/ExtractThinker/issues/287).
4. **Predictable operations.** Bounded concurrency, documented cancellation behavior, consistent timeout units, usage/cost metadata when providers supply it, structured logs with document contents excluded by default. Existing `extract_async` uses `asyncio.to_thread`; document that behavior before promising native async streaming.
5. **Agent integration after reliability.** A small optional CLI/MCP integration is a candidate from [#252](https://github.com/enoch3712/ExtractThinker/issues/252), not the initial release blocker. Keep the Python library usable independently.

Defer a hosted platform, many new loaders, an extensive UI application and blanket performance claims until the core journey is proven.

## Visual and documentation facelift

Recommended direction: retain recognizable pink as a restrained accent, with ink-colored text, warm light surfaces and a properly supported dark theme. Use a document-to-typed-object motif consistently in the logo refinement, diagrams, README banner and social preview. Keep illustrations subordinate to readable code and real output.

Homepage composition:

1. Hero: “Turn complex documents into validated Python objects.” Supporting sentence, **Get started** and **View on GitHub**.
2. Side-by-side sample invoice, a short contract and resulting JSON. Highlight source-to-field correspondence only where supported; otherwise label it explanatory artwork.
3. Three workflow cards: extract an invoice; classify and split a mixed packet; process using a local model.
4. A compact ecosystem row showing tested integrations, linking to capability/version guidance.
5. Reproducible evaluation results with corpus, model, date, failure rate, latency and cost methodology. No invented accuracy number.
6. Contributor invitation and recent release notes.

README: outcome → short demo → verified install/quickstart → why this library → recipes → supported integrations → contribution/license. Replace the current long feature tour with links to task-oriented guides.

Docs navigation: **Start here / Recipes / Concepts / Integrations / API reference / Migration & releases / Contributing**. Split getting-started paths for hosted LLMs and local processing. Make optional system dependencies discoverable before users hit import errors.

Retain MkDocs initially; a framework migration does not itself improve extraction or onboarding. Acceptance criteria: useful HTML without the homepage redirect, no placeholder navigation, visible keyboard focus, readable contrast, mobile navigation, correct code-copy behavior, accessible light/dark palettes, and verified internal links. Capture desktop and mobile previews before publishing.

## Ecosystem positioning

[Docling](https://github.com/docling-project/docling) focuses on document conversion; [Unstructured](https://github.com/Unstructured-IO/unstructured) offers document transformation/ETL tooling; [Instructor](https://github.com/567-labs/instructor) centers on structured LLM outputs. These official project descriptions support the following positioning inference: ExtractThinker can distinguish itself through the complete classify → split → contract extraction workflow, while composing with parsers and structured-output libraries.

Publish a factual “when to use which” guide with runnable examples. Avoid unsupported superiority or speed claims. Partner tutorials with existing integrations are more credible than presenting every adjacent project as a competitor.

## Backlog triage before new development

Review all 29 issues and 5 PRs. Preserve attribution and prefer improving existing contributions over duplicating them. Do not automatically merge or close anything based on this audit.

- [PR #362](https://github.com/enoch3712/ExtractThinker/pull/362): review stability/logging/temp-directory work first; test cleanup, concurrency and sensitive logging.
- [PR #353](https://github.com/enoch3712/ExtractThinker/pull/353): evaluate with the concatenation regressions above.
- [PR #363](https://github.com/enoch3712/ExtractThinker/pull/363): verify what a first-class MiniMax integration adds beyond existing provider support and require adapter tests.
- [PR #335](https://github.com/enoch3712/ExtractThinker/pull/335): reuse useful architecture decisions in contributor documentation.
- [PR #72](https://github.com/enoch3712/ExtractThinker/pull/72): review entity masking separately; clarify its limits and compatibility before adding public guarantees.
- [#309](https://github.com/enoch3712/ExtractThinker/issues/309): reproduce Docling import compatibility against the supported integration versions.
- [#141](https://github.com/enoch3712/ExtractThinker/issues/141): include split/classification correctness in the regression corpus.

Use labels for reproducible bug, needs reproduction, documentation, integration, and proposed feature. Assign an owner and next action to every accepted relaunch item.

## Delivery plan: roughly 12 weeks from kickoff

Assumption: one active maintainer with targeted community help. Dates are planning estimates, not a commitment; move the launch if the release gates fail.

| Window | Deliverable | Exit gate |
| --- | --- | --- |
| Weeks 1–2 | Backlog triage; wheel-install quickstart; deterministic PR CI; support-policy decision | A new contributor can install and run offline checks without maintainer credentials; priority bugs have reproductions |
| Weeks 3–4 | Dependency batches; critical fixes; integration matrix; patch release if compatible | Tested wheel/sdist, supported-Python jobs, clean lock check, documented migrations, provider smoke tests on declared configurations |
| Weeks 5–6 | README, branding, homepage and recipe-led docs; three complete recipes | Strict docs build, mobile/desktop review, tested examples, public limitations and release notes |
| Weeks 7–8 | Reproducible evaluation harness and relaunch candidate | Versioned corpus with permission to distribute; accuracy/completeness/failure metrics; measured latency and token/cost reporting |
| Weeks 9–12 | Relaunch, integration tutorials, issue sprints and contributor follow-through | Maintainer response cadence is sustainable; evidence shows users completing workflows; next cycle chosen from user feedback |

Suggested small PRs, in order: **01** tested quickstart; **02** offline CI and test taxonomy; **03** contributor tooling; **04** dependency/support-policy batches; **05** long-document regressions and fixes, incorporating existing PRs; **06** release provenance and changelog; **07** README and visual/docs redesign; **08** recipes/evaluations; **09** optional agent integration after launch gates.

## Route to 10k

The gap is 8,403 stars. At an assumed net gain of 250 / 500 / 1,000 stars monthly, the arithmetic is about 34 / 17 / 9 months. These are scenarios, not forecasts; no growth history or funnel baseline was measured in this audit.

| Milestone | Evidence to earn it | Distribution experiment |
| --- | --- | --- |
| 2k | Maintained release, working quickstart, backlog response | Relaunch update and 60-second real demo |
| 3k | Three excellent end-to-end recipes and comparison guide | Invoice processing, mixed-document packets and local-model tutorials |
| 5k | Reproducible evaluations and recurring outside contributions | Integration collaborations, user case studies and contributor sprints |
| 10k | Sustained usage, reliable releases and multiple maintainers | Repeat the channels that bring retained users and contributors |

Record a baseline and review monthly: net stars, docs traffic and quickstart clicks where measurement is available, package downloads with bot/mirror caveats, first-response time, issue age, unique contributors and repeat contributors. Track successful first runs through opt-in feedback/usability sessions rather than silently adding library telemetry. Do not equate downloads or stars with active users.

Proposed operating targets: acknowledge new issues within three working days; review actionable PRs weekly; publish a monthly maintenance update; ship only when release checks pass. Revisit these targets against available maintainer time.

Use focused launch posts and demonstrations in relevant developer communities, honoring their posting rules. Publish one useful tutorial per week during the launch window and request feedback from existing contributors. Do not buy stars, automate promotional messages, or optimize for attention that does not produce successful users.

## Validation performed and limits

- Fetch succeeded; fresh worktree matches current `origin/main`.
- All 67 library Python files parsed successfully with the local Python 3.14.4 interpreter. Syntax parsing is not runtime compatibility validation.
- All 48 Markdown destinations in MkDocs navigation exist; placeholder links remain a separate finding.
- `poetry check --lock` passed with metadata-deprecation warnings. This does not establish dependency/runtime compatibility.
- GitHub metadata, issues, releases and recent Actions history were inspected; package candidate versions were queried from PyPI.
- Full runtime tests, provider calls, a clean installation, vulnerability audit, docs rendering and benchmark runs were not performed. No claim of a passing current runtime suite is made.
- This pass adds a plan only. Dependencies, production code, public docs, repository settings, releases and external conversations were not changed. Original local work remains intact.

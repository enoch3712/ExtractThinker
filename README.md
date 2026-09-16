<p align="center"><img src="docs/assets/logomain.png" alt="ExtractThinker" width="240"></p>

# Complex documents. Validated Python objects.

ExtractThinker is an open-source Python library for turning documents into typed data. Define a Pydantic contract, choose a document parser and LLM, then load, classify, split and extract.

[Documentation](https://enoch3712.github.io/ExtractThinker/) · [Quickstart](https://enoch3712.github.io/ExtractThinker/getting-started/) · [Contributing](CONTRIBUTING.md) · [2026 roadmap](planning/2026-facelift.md)

![Python 3.9–3.13](https://img.shields.io/badge/Python-3.9–3.13-blue)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Last commit](https://img.shields.io/github/last-commit/enoch3712/ExtractThinker)](https://github.com/enoch3712/ExtractThinker/commits/main)

## First extraction

```bash
pip install extract-thinker
```

Set `EXTRACT_THINKER_MODEL` to a model available to you and configure that provider's API key. Save this sample as `invoice.txt`:

```text
Invoice: INV-2026-001
Supplier: Example Company
Total: 120.00 EUR
```

```python
import os
from pydantic import Field
from extract_thinker import Contract, DocumentLoaderTxt, Extractor, LLM

class Invoice(Contract):
    invoice_number: str
    supplier: str
    total: float = Field(ge=0)
    currency: str

extractor = Extractor(
    DocumentLoaderTxt(),
    LLM(os.environ["EXTRACT_THINKER_MODEL"], token_limit=1000),
)
result = extractor.extract("invoice.txt", Invoice)
print(result.model_dump())
```

The result is a validated `Invoice`. Schema validity does not guarantee factual accuracy; evaluate results on your documents. Extraction uses the configured provider and may incur charges.

For PDFs, install `pypdf` and use `DocumentLoaderPyPdf`. Scanned documents need OCR or a vision-capable model. System MIME detection requires libmagic (`brew install libmagic` on macOS or `apt-get install libmagic1` on Debian/Ubuntu). See the [quickstart](https://enoch3712.github.io/ExtractThinker/getting-started/) for a credential-free loading check and setup details.

## Build a document workflow

| Need | Component |
| --- | --- |
| Read PDFs, images, tables and spreadsheets | [Document loaders](https://enoch3712.github.io/ExtractThinker/core-concepts/document-loaders/) |
| Define fields, constraints and post-validation | [Pydantic contracts](https://enoch3712.github.io/ExtractThinker/core-concepts/contracts/) |
| Classify and split mixed document bundles | [Classification](https://enoch3712.github.io/ExtractThinker/core-concepts/classification/) and [splitters](https://enoch3712.github.io/ExtractThinker/core-concepts/splitters/) |
| Handle long inputs and incomplete responses | [Completion strategies](https://enoch3712.github.io/ExtractThinker/core-concepts/completion-strategies/) |
| Use local models | [Ollama setup](https://enoch3712.github.io/ExtractThinker/core-concepts/llm-integration/ollama/) |

## 2026 additions on main

The repository now includes page retrieval with SQLite, parallel field extraction, configurable model routing, local entity masking, page events, PyMuPDF/Camelot/Tabula/Adobe loaders, and an MCP service with Docker Compose. These changes are **not yet a new PyPI release**. Install from a checkout to use them:

```bash
git clone https://github.com/enoch3712/ExtractThinker.git
cd ExtractThinker
pip install -e .
```

Read the [2026 update and compatibility notes](docs/getting-started/2026-update.md), [MCP setup](docs/getting-started/mcp-container.md), and [issue-resolution evidence](planning/issue-resolution.md). The core supports Python 3.9–3.13; the optional MCP service requires Python 3.10+.

## Contribute

Start with [CONTRIBUTING.md](CONTRIBUTING.md). The offline core suite runs without provider credentials. Good contributions include reduced document fixtures, loader compatibility fixes, examples with expected outputs, and clear reports of model or parser limitations.

If ExtractThinker helps your project, a GitHub star helps others find it. The [roadmap](planning/2026-facelift.md) focuses on reliable onboarding, reproducible examples and contributor support.

## Articles and project history

Stay updated and connect with the community:
- [Scaling Document Extraction with o1, GPT-4o & Mini](https://medium.com/towards-artificial-intelligence/scaling-document-extraction-with-o1-gpt4o-and-mini-extractthinker-8f3340b4e69c)
- [Claude 3.5 — The King of Document Intelligence](https://medium.com/gitconnected/claude-3-5-the-king-of-document-intelligence-f57bea1d209d?sk=124c5abb30c0e7f04313c5e20e79c2d1)
- [Classification Tree for LLMs](https://medium.com/gitconnected/classification-tree-for-llms-32b69015c5e0?sk=8a258cf74fe3483e68ab164e6b3aaf4c)
- [Advanced Document Classification with LLMs](https://medium.com/gitconnected/advanced-document-classification-with-llms-8801eaee3c58?sk=f5a22ee72022eb70e112e3e2d1608e79)
- [Phi-3 and Azure: PDF Data Extraction | ExtractThinker](https://medium.com/towards-artificial-intelligence/phi-3-and-azure-pdf-data-extraction-extractthinker-cb490a095adb?sk=7be7e625b8f9932768442f87dd0ebcec)
- [ExtractThinker: Document Intelligence for LLMs](https://medium.com/towards-artificial-intelligence/extractthinker-ai-document-intelligence-with-llms-72cbce1890ef)

## License

[Apache License 2.0](LICENSE).

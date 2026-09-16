# Contributing to ExtractThinker

Use a Python version declared in `pyproject.toml` (currently 3.9–3.13).
Create a virtual environment, then install the project and offline test dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e . -r requirements-test.txt
python -m pytest tests/unit tests/test_document_loader_data.py tests/test_document_loader_pypdf.py tests/test_document_loader_txt.py -q
```

On Windows, activate with `.venv\Scripts\activate`. MIME detection requires the
system libmagic library (`brew install libmagic` on macOS,
`sudo apt-get install libmagic1` on Debian/Ubuntu).

The offline suite exercises local loaders and provider adapters without
API credentials. Add deterministic regressions under `tests/unit` for bugs.
Keep original provider responses and document fixtures free of private data.
Assert behavior rather than exact LLM wording or elapsed-time improvements.

The remaining tests include optional integrations, OCR model downloads,
local model servers and cloud API calls. Install the loader's optional packages
and configure credentials only for the integration you intend to exercise.
Run `python -m pytest tests/critical/ -v` explicitly with `GROQ_API_KEY` configured
for the existing critical Groq tests. A passing offline suite does not
claim live-provider compatibility.

For documentation:

```bash
python -m pip install -r requirements-docs.txt
python -m mkdocs build --strict
```

Describe the user-visible change, link the issue, and list actual validation in
your pull request. Update affected documentation and note compatibility changes.
Do not close an issue solely because a test passes: verify the requested behavior
and link to the delivered fix. New integrations should stay optional and include
installation guidance and deterministic adapter tests.

For the optional MCP suite (Python 3.10+):

```bash
python -m pip install -r requirements-server.txt
python -m pytest tests/test_mcp_server.py -q
```

The MCP tests use the real SDK with a fake provider transport. To verify the
container, build it with `docker compose build`, start it, then check `/healthz`
and connect an MCP client to `/mcp`. Do not put provider keys in committed files.

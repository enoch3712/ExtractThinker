# MCP service and Docker container

Run ExtractThinker as an MCP service so a client can discover local documents and extract them into a JSON Schema contract. The service supports PDF, UTF-8 text/Markdown, PNG, JPEG and WebP. It exposes `server_info`, `list_documents` and `extract_document` over Streamable HTTP or stdio.

## Start with Docker Compose

From a checkout of this repository:

```bash
cp server.env.example server.env
# Edit server.env: set EXTRACT_THINKER_MODEL and that provider's API key.
docker compose up --build -d
curl http://127.0.0.1:8000/healthz
```

Connect an MCP client to `http://127.0.0.1:8000/mcp`. The default mount contains the sample invoice in `examples/service-data`. Set `EXTRACT_THINKER_DOCUMENTS` to an absolute directory before starting Compose to use your documents:

```bash
EXTRACT_THINKER_DOCUMENTS=/absolute/path/to/documents docker compose up --build -d
```

The image is built locally; no registry image is required. It runs as a non-root user with a read-only document mount and filesystem, plus a temporary directory. Ensure the mounted documents are readable by UID 10001. `server.env` is ignored by Git and excluded from the image. Stop the service with `docker compose down`.

Compose binds only to localhost. The service does not implement authentication: use an authenticated reverse proxy and appropriate access controls before exposing it to other machines. The configured model receives selected document content, including images when vision is enabled.

## Extract a document

For a Python client, install `mcp>=2.2,<3` (Python 3.10 or newer):

```python
import asyncio
from mcp import Client

async def main():
    async with Client("http://127.0.0.1:8000/mcp") as client:
        result = await client.call_tool("extract_document", {
            "path": "invoice.txt",
            "contract_schema": {
                "type": "object",
                "properties": {"total": {"type": "number"}},
                "required": ["total"],
                "additionalProperties": False,
            },
        })
        if result.is_error:
            raise RuntimeError(result.content)
        print(result.structured_content)
        # {"data": {"total": 120}, "pages": [1], "model": "..."}

asyncio.run(main())
```

Paths must be relative to the mounted root; paths and symlinks outside it are rejected. `list_documents` returns up to 200 filenames and sizes. The server model is configured by the operator, rather than chosen by each request.

Optional extraction arguments:

- `pages`: a unique list of one-based source page numbers, such as `[2, 5]`. PDF pages are selected before rendering; results retain those source numbers.
- `vision`: enable for image files or scanned PDFs, using a vision-capable model. Text extraction alone does not OCR scanned PDFs.
- `instructions`: additional extraction instructions.

Contracts use JSON Schema Draft 2020-12, must describe an object, and are limited to 64 KiB. Local `$ref` definitions are supported; remote references are rejected. Provider responses are validated against the supplied contract. Contracts do not execute Python validators.

Default limits are 20 MiB per file, 100 selected pages and 4,096 output tokens. Configure these through the variables in `server.env.example`. These are service limits, not a guarantee that every model can fit the selected document in its context window. Select fewer pages for large documents. Encrypted PDFs are not supported by this service.

## Run without Docker

The core library supports Python 3.9; this optional service requires Python 3.10+ and uses the MCP 2.x SDK. From the repository root:

```bash
pip install . -r requirements-server.txt
export EXTRACT_THINKER_DOCUMENT_ROOT=/absolute/path/to/documents
export EXTRACT_THINKER_MODEL=your-provider/your-model
# Configure credentials for your provider.
python -m extract_thinker.mcp_server
```

For a client that launches an stdio subprocess, use the same environment and command with `--transport stdio`. HTTP host and port can be set with `--host` and `--port`. `/healthz` checks process availability and reports whether a model name is configured; it does not test provider credentials or model availability.

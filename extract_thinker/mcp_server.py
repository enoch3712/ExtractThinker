"""Optional MCP service. Install requirements-server.txt before running."""
import argparse
import asyncio
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, RootModel, model_validator
from extract_thinker.extractor import Extractor
from extract_thinker.llm import LLM
from extract_thinker.document_loader.document_loader_data import DocumentLoaderData
from extract_thinker.document_loader.document_loader_llm_image import DocumentLoaderLLMImage
from extract_thinker.document_loader.document_loader_pypdf import DocumentLoaderPyPdf

SUPPORTED_SUFFIXES = {'.pdf', '.txt', '.md', '.png', '.jpg', '.jpeg', '.webp'}


@dataclass
class ServiceConfig:
    document_root: Path
    model: Optional[str] = None
    max_file_bytes: int = 20 * 1024 * 1024
    max_pages: int = 100
    output_tokens: int = 4096

    def __post_init__(self):
        self.document_root = Path(self.document_root).resolve(strict=True)
        if not self.document_root.is_dir():
            raise ValueError('document_root must be a directory')
        for name in ('max_file_bytes', 'max_pages', 'output_tokens'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')

    @classmethod
    def from_environment(cls):
        return cls(
            document_root=Path(os.getenv('EXTRACT_THINKER_DOCUMENT_ROOT', './documents')),
            model=os.getenv('EXTRACT_THINKER_MODEL') or None,
            max_file_bytes=int(os.getenv('EXTRACT_THINKER_MAX_FILE_BYTES', str(20 * 1024 * 1024))),
            max_pages=int(os.getenv('EXTRACT_THINKER_MAX_PAGES', '100')),
            output_tokens=int(os.getenv('EXTRACT_THINKER_OUTPUT_TOKENS', '4096')),
        )


def contract_from_schema(schema):
    """Validate a JSON object contract without generating or executing Python."""
    from jsonschema import Draft202012Validator
    if not isinstance(schema, dict) or schema.get('type') != 'object':
        raise ValueError('contract_schema must be a JSON Schema with type=object')
    if len(json.dumps(schema).encode('utf-8')) > 65536:
        raise ValueError('contract_schema exceeds 64 KiB')
    def check_references(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ('$ref', '$dynamicRef') and (not isinstance(item, str) or not item.startswith('#')):
                    raise ValueError('Only local JSON Schema references are supported')
                check_references(item)
        elif isinstance(value, list):
            for item in value:
                check_references(item)
    check_references(schema)
    Draft202012Validator.check_schema(schema)
    contract_schema = deepcopy(schema)
    validator = Draft202012Validator(contract_schema)

    class JSONContract(RootModel[Dict[str, Any]]):
        @classmethod
        def model_json_schema(cls, *args, **kwargs):
            return deepcopy(contract_schema)

        @model_validator(mode='before')
        @classmethod
        def validate_contract(cls, value):
            errors = list(validator.iter_errors(value))
            if errors:
                raise ValueError(f'Response violates contract: {errors[0].message}')
            return value

    return JSONContract


class ExtractionService:
    def __init__(self, config, llm_factory=LLM):
        self.config = config
        self.llm_factory = llm_factory

    def info(self):
        return {'name': 'ExtractThinker', 'version': version('extract_thinker'),
                'model_configured': bool(self.config.model),
                'supported_extensions': sorted(SUPPORTED_SUFFIXES),
                'max_file_bytes': self.config.max_file_bytes, 'max_pages': self.config.max_pages}

    def _path(self, relative_path):
        candidate = Path(relative_path)
        if candidate.is_absolute():
            raise ValueError('Document paths must be relative to the document root')
        path = (self.config.document_root / candidate).resolve(strict=True)
        if not path.is_relative_to(self.config.document_root):
            raise ValueError('Document path is outside the configured root')
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError('Unsupported document format')
        return path

    def list_documents(self):
        documents = []
        for path in sorted(self.config.document_root.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            relative = path.relative_to(self.config.document_root).as_posix()
            try:
                resolved = self._path(relative)
            except (ValueError, OSError):
                continue
            documents.append({'path': relative, 'bytes': resolved.stat().st_size})
            if len(documents) == 200:
                break
        return documents

    def extract(self, path, contract_schema, vision=False, pages=None, instructions=None):
        if not self.config.model:
            raise ValueError('Set EXTRACT_THINKER_MODEL and the provider credentials before extracting')
        contract = contract_from_schema(contract_schema)
        resolved = self._path(path)
        with resolved.open('rb') as source:
            data = source.read(self.config.max_file_bytes + 1)
        if len(data) > self.config.max_file_bytes:
            raise ValueError('Document exceeds the configured byte limit')
        suffix = resolved.suffix.lower()
        if suffix == '.pdf':
            from pypdf import PdfReader, PdfWriter
            reader = PdfReader(BytesIO(data))
            if reader.is_encrypted:
                raise ValueError('The service requires an unencrypted PDF')
            total = len(reader.pages)
            selected = list(range(1, total + 1)) if pages is None else pages
            self._validate_pages(selected, total)
            # Select before rendering or extraction, including for large PDFs.
            writer = PdfWriter()
            for number in selected:
                writer.add_page(reader.pages[number - 1])
            stream = BytesIO()
            writer.write(stream)
            loader = DocumentLoaderPyPdf(vision_enabled=vision)
            loader.set_max_image_size(1024)
            loaded = loader.load(stream)
        else:
            selected = [1] if pages is None else pages
            self._validate_pages(selected, 1)
            if suffix in ('.txt', '.md'):
                if vision:
                    raise ValueError('Text files do not provide vision input')
                loaded = [{'content': data.decode('utf-8-sig')}]
            else:
                if not vision:
                    raise ValueError('Image files require vision=True')
                loader = DocumentLoaderLLMImage(max_image_size=1024)
                loaded = loader.load(BytesIO(data))
        if len(loaded) != len(selected):
            raise ValueError('Loader did not return all selected pages')
        loaded = [dict(page, page_number=number) for page, number in zip(loaded, selected)]
        llm = self.llm_factory(self.config.model, token_limit=self.config.output_tokens)
        extractor = Extractor(DocumentLoaderData(), llm)
        result = extractor.extract(loaded, contract, vision=vision, content=instructions)
        output = result.model_dump()
        contract.model_validate(output)
        return {'data': output, 'pages': selected, 'model': self.config.model}

    def _validate_pages(self, pages, total):
        if not isinstance(pages, list) or not pages or len(pages) > self.config.max_pages:
            raise ValueError('Select between one and max_pages pages')
        if any(isinstance(page, bool) or not isinstance(page, int) or not 1 <= page <= total for page in pages):
            raise ValueError('Invalid one-based page selection')
        if len(set(pages)) != len(pages):
            raise ValueError('Page selection contains duplicates')


class ExtractionResult(BaseModel):
    data: Dict[str, Any]
    pages: List[int]
    model: str


def create_server(config=None, llm_factory=LLM):
    from mcp.server import MCPServer
    from starlette.responses import JSONResponse
    service = ExtractionService(config or ServiceConfig.from_environment(), llm_factory)
    server = MCPServer('ExtractThinker', version=version('extract_thinker'),
                       instructions='Extract structured data from documents inside the configured document root.')

    @server.tool(structured_output=True)
    def server_info() -> Dict[str, Any]:
        """Return supported formats, limits and model configuration status."""
        return service.info()

    @server.tool(structured_output=True)
    def list_documents() -> List[Dict[str, Any]]:
        """List up to 200 supported documents in the configured root."""
        return service.list_documents()

    @server.tool(structured_output=True)
    async def extract_document(path: str, contract_schema: Dict[str, Any], vision: bool = False,
                               pages: Optional[List[int]] = None, instructions: Optional[str] = None) -> ExtractionResult:
        """Extract a local document into an object matching a JSON Schema contract.

        Paths are relative to the document root. Images require vision=True.
        Pages are one-based. The configured model receives the selected content.
        """
        result = await asyncio.to_thread(service.extract, path, contract_schema, vision, pages, instructions)
        return ExtractionResult(**result)

    @server.custom_route('/healthz', methods=['GET'])
    async def health(request):
        return JSONResponse({'status': 'ok', **service.info()})

    return server


def main():
    parser = argparse.ArgumentParser(description='Run the ExtractThinker MCP service')
    parser.add_argument('--transport', choices=['stdio', 'streamable-http'], default='streamable-http')
    parser.add_argument('--host', default=os.getenv('EXTRACT_THINKER_HOST', '127.0.0.1'))
    parser.add_argument('--port', type=int, default=int(os.getenv('EXTRACT_THINKER_PORT', '8000')))
    args = parser.parse_args()
    server = create_server()
    if args.transport == 'stdio':
        server.run('stdio')
    else:
        server.run('streamable-http', host=args.host, port=args.port, stateless_http=True, json_response=True)


if __name__ == '__main__':
    main()

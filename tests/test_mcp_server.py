"""Optional MCP runtime tests with a fake LLM; no provider requests."""
import asyncio
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest
pytest.importorskip('mcp')
from mcp import Client
from pypdf import PdfWriter
from PIL import Image
from extract_thinker.mcp_server import ServiceConfig, ExtractionService, contract_from_schema, create_server

SCHEMA = {'type': 'object', 'properties': {'amount': {'type': 'integer', 'minimum': 1}},
          'required': ['amount'], 'additionalProperties': False}


class FakeLLM:
    model = 'test/model'
    def __init__(self, model, token_limit=None):
        self.model = model
        self.last_completion = None
    def set_page_count(self, count):
        pass
    def request(self, messages, response_model):
        return response_model.model_validate({'amount': 120})


def test_contract_validation_supports_local_refs_and_no_code_generation():
    schema = {'type': 'object', 'properties': {'amount': {'$ref': '#/$defs/positive'}},
              '$defs': {'positive': {'type': 'integer', 'minimum': 1}}, 'required': ['amount']}
    model = contract_from_schema(schema)
    assert model.model_json_schema() == schema
    assert model.model_validate({'amount': 120}).model_dump() == {'amount': 120}
    with pytest.raises(ValueError):
        model.model_validate({'amount': 0})
    with pytest.raises(ValueError, match='local JSON Schema references'):
        contract_from_schema({'type': 'object', '$ref': 'https://example.invalid/schema'})


def test_text_extraction_and_document_listing(tmp_path):
    (tmp_path / 'invoice.txt').write_text('Total 120')
    service = ExtractionService(ServiceConfig(tmp_path, model='test/model'), FakeLLM)
    assert service.list_documents() == [{'path': 'invoice.txt', 'bytes': 9}]
    assert service.extract('invoice.txt', SCHEMA) == {'data': {'amount': 120}, 'pages': [1], 'model': 'test/model'}


def test_path_escape_and_symlink_are_rejected(tmp_path):
    outside = tmp_path.parent / 'outside-mcp.txt'
    outside.write_text('outside')
    root = tmp_path / 'root'
    root.mkdir()
    (root / 'link.txt').symlink_to(outside)
    service = ExtractionService(ServiceConfig(root, model='test/model'), FakeLLM)
    assert service.list_documents() == []
    with pytest.raises(ValueError, match='outside'):
        service.extract('link.txt', SCHEMA)
    with pytest.raises(ValueError, match='relative'):
        service.extract(str(outside), SCHEMA)


def test_pdf_page_limit_and_selection_happen_before_extraction(tmp_path):
    writer = PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=100, height=100)
    writer.write(tmp_path / 'file.pdf')
    service = ExtractionService(ServiceConfig(tmp_path, model='test/model', max_pages=1), FakeLLM)
    with pytest.raises(ValueError, match='max_pages'):
        service.extract('file.pdf', SCHEMA)
    assert service.extract('file.pdf', SCHEMA, pages=[3])['pages'] == [3]
    with pytest.raises(ValueError, match='page selection'):
        service.extract('file.pdf', SCHEMA, pages=[4])


def test_byte_limit_and_missing_model(tmp_path):
    (tmp_path / 'file.txt').write_text('large content')
    with pytest.raises(ValueError, match='byte limit'):
        ExtractionService(ServiceConfig(tmp_path, model='test/model', max_file_bytes=2), FakeLLM).extract('file.txt', SCHEMA)
    with pytest.raises(ValueError, match='EXTRACT_THINKER_MODEL'):
        ExtractionService(ServiceConfig(tmp_path), FakeLLM).extract('file.txt', SCHEMA)


@pytest.mark.parametrize('format', ['PNG', 'WEBP', 'JPEG'])
def test_image_formats_and_vision(tmp_path, format):
    suffix = 'jpg' if format == 'JPEG' else format.lower()
    path = tmp_path / f'image.{suffix}'
    Image.new('RGB', (10, 10)).save(path, format=format)
    service = ExtractionService(ServiceConfig(tmp_path, model='test/model'), FakeLLM)
    with pytest.raises(ValueError, match='vision=True'):
        service.extract(path.name, SCHEMA)
    assert service.extract(path.name, SCHEMA, vision=True)['data'] == {'amount': 120}


def test_mcp_client_tool_roundtrip(tmp_path):
    (tmp_path / 'invoice.txt').write_text('Total 120')
    server = create_server(ServiceConfig(tmp_path, model='test/model'), FakeLLM)
    async def run():
        async with Client(server) as client:
            tools = await client.list_tools()
            assert {tool.name for tool in tools.tools} == {'server_info', 'list_documents', 'extract_document'}
            result = await client.call_tool('extract_document', {'path': 'invoice.txt', 'contract_schema': SCHEMA})
            assert not result.is_error
            assert result.structured_content['data'] == {'amount': 120}
    asyncio.run(run())


def test_real_instructor_transport_receives_and_validates_schema(tmp_path, monkeypatch):
    import json
    from litellm import ModelResponse
    from extract_thinker import LLM
    schema = {'type': 'object', 'properties': {'amount': {'$ref': '#/$defs/positive'}},
              '$defs': {'positive': {'type': 'integer', 'minimum': 1}}, 'required': ['amount']}
    completion = Mock(return_value=ModelResponse(
        model='test-model', choices=[{'index': 0, 'finish_reason': 'stop',
        'message': {'role': 'assistant', 'content': '{"amount":120}'}}]))
    monkeypatch.setattr('extract_thinker.llm.litellm.completion', completion)
    (tmp_path / 'invoice.txt').write_text('Total 120')
    service = ExtractionService(ServiceConfig(tmp_path, model='test-model'), LLM)
    assert service.extract('invoice.txt', schema)['data'] == {'amount': 120}
    prompt = json.dumps(completion.call_args.kwargs['messages'])
    assert '$defs' in prompt and 'minimum' in prompt

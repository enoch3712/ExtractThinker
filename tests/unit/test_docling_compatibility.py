from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import extract_thinker.document_loader.document_loader_docling as adapter
from extract_thinker import DocumentLoaderDocling, DoclingConfig
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader


def test_missing_docling_has_install_instruction(monkeypatch):
    error = ModuleNotFoundError("No module named 'docling'", name="docling")
    monkeypatch.setattr(adapter, 'import_module', Mock(side_effect=error))
    with pytest.raises(ImportError, match='Docling is optional') as caught:
        DoclingConfig()
    assert caught.value.__cause__ is error


def test_inconsistent_core_preserves_root_cause_and_repair(monkeypatch):
    error = ModuleNotFoundError("No module named 'docling_core.types.doc.page'",
                                name='docling_core.types.doc.page')
    monkeypatch.setattr(adapter, 'import_module', Mock(side_effect=error))
    monkeypatch.setattr(adapter, 'version', lambda package: '2.0.0')
    with pytest.raises(ImportError, match='upgrade-strategy eager docling') as caught:
        DocumentLoaderDocling()
    assert 'docling-core=2.0.0' in str(caught.value)
    assert 'docling_core.types.doc.page' in str(caught.value)
    assert caught.value.__cause__ is error


def loader_with_result(monkeypatch, page_numbers):
    loader = object.__new__(DocumentLoaderDocling)
    CachedDocumentLoader.__init__(loader)
    document = SimpleNamespace(export_to_markdown=Mock(side_effect=lambda **kw: f"page {kw.get('page_no', 'all')}"))
    result = SimpleNamespace(pages=[SimpleNamespace(page_no=n) for n in page_numbers], document=document)
    monkeypatch.setattr(loader, '_docling_convert', Mock(return_value=result))
    return loader, document


def test_one_based_page_filter_and_images_rendered_once(monkeypatch):
    loader, document = loader_with_result(monkeypatch, [1, 2])
    loader.set_vision_mode(True)
    render = Mock(return_value={0: b'first', 1: b'second'})
    monkeypatch.setattr(loader, 'convert_to_images', render)
    pages = loader.load('file.pdf')
    assert [p['content'] for p in pages] == ['page 1', 'page 2']
    assert [p['markdown'] for p in pages] == ['page 1', 'page 2']
    assert [p['image'] for p in pages] == [b'first', b'second']
    assert [p['page_number'] for p in pages] == [1, 2]
    render.assert_called_once_with('file.pdf')


def test_remote_pdf_keeps_individual_pages(monkeypatch):
    loader, _ = loader_with_result(monkeypatch, [1, 2])
    assert len(loader.load('https://example.org/file.pdf')) == 2


def test_unpaginated_document_has_markdown(monkeypatch):
    loader, _ = loader_with_result(monkeypatch, [])
    assert loader.load('file.html') == [dict(content='page all', markdown='page all', page_number=1, image=None)]


def test_existing_converter_reused_and_stream_position_preserved(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'docling_core.types.io', SimpleNamespace(DocumentStream=lambda **kw: SimpleNamespace(**kw)))
    loader = object.__new__(DocumentLoaderDocling)
    loader.converter = SimpleNamespace(convert=Mock(return_value='result'))
    source = BytesIO(b'PDF content')
    source.seek(4)
    assert loader._docling_convert(source) == 'result'
    argument = loader.converter.convert.call_args.args[0]
    assert argument.stream.read() == b'PDF content'
    assert source.tell() == 4
    assert loader._docling_convert('file.pdf') == 'result'
    assert loader.converter.convert.call_count == 2

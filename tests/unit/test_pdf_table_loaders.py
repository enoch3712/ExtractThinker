import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pymupdf
import pytest
from extract_thinker import DocumentLoaderCamelot, CamelotConfig, DocumentLoaderTabula, TabulaConfig


@pytest.fixture
def pdf_source(tmp_path):
    path = tmp_path / 'tables.pdf'
    with pymupdf.open() as pdf:
        for content in ('Table page', '', 'Another table'):
            page = pdf.new_page()
            if content:
                page.insert_text((40, 40), content)
        pdf.save(path)
    return path


class Frame:
    def fillna(self, value):
        return self
    def astype(self, value):
        return self
    @property
    def values(self):
        return self
    def tolist(self):
        return [['Name', 'Amount'], ['Alice', ''], ['Bob', '10']]


def test_camelot_preserves_blank_pages_and_table_boundaries(monkeypatch, pdf_source):
    read = Mock(return_value=[SimpleNamespace(page=3, df=Frame()), SimpleNamespace(page=1, df=Frame()), SimpleNamespace(page=3, df=Frame())])
    monkeypatch.setitem(sys.modules, 'camelot', SimpleNamespace(read_pdf=read))
    loader = DocumentLoaderCamelot(CamelotConfig(flavor='stream', read_pdf_kwargs={'split_text': True}))
    source = BytesIO(pdf_source.read_bytes())
    source.seek(6)
    pages = loader.load(source)
    assert [len(page['tables']) for page in pages] == [1, 0, 2]
    assert pages[0]['tables'][0][1] == ['Alice', '']
    assert pages[2]['page_number'] == 3
    assert 'Table page' in pages[0]['content']
    assert source.tell() == 6
    assert not Path(read.call_args.args[0]).exists()
    assert read.call_args.kwargs['pages'] == 'all'
    assert read.call_args.kwargs['split_text'] is True
    assert loader.load(source) == pages
    read.assert_called_once()


def test_tabula_requests_each_page_and_keeps_empty_cells(monkeypatch, pdf_source):
    raw = {'data': [[{'text': 'Item'}, {'text': 'Price'}], [{'text': 'Book'}, {'text': ''}]]}
    read = Mock(side_effect=[[raw], [], [raw, raw]])
    monkeypatch.setitem(sys.modules, 'tabula', SimpleNamespace(read_pdf=read))
    loader = DocumentLoaderTabula(TabulaConfig(lattice=True))
    pages = loader.load(str(pdf_source))
    assert [len(page['tables']) for page in pages] == [1, 0, 2]
    assert pages[2]['tables'][0][1] == ['Book', '']
    assert [call.kwargs['pages'] for call in read.call_args_list] == [1, 2, 3]
    assert all(call.kwargs['output_format'] == 'json' for call in read.call_args_list)


def test_failure_cleans_temporary_pdf_and_does_not_cache(monkeypatch, pdf_source):
    paths = []
    def fail(path, **kwargs):
        paths.append(path)
        assert Path(path).exists()
        raise RuntimeError('table parser failed')
    monkeypatch.setitem(sys.modules, 'camelot', SimpleNamespace(read_pdf=fail))
    loader = DocumentLoaderCamelot()
    with pytest.raises(RuntimeError, match='table parser failed'):
        loader.load(BytesIO(pdf_source.read_bytes()))
    assert not Path(paths[0]).exists()
    assert len(loader.cache) == 0


def test_vision_and_page_selection(monkeypatch, pdf_source):
    monkeypatch.setitem(sys.modules, 'camelot', SimpleNamespace(read_pdf=Mock(return_value=[])))
    loader = DocumentLoaderCamelot(CamelotConfig(vision_enabled=True))
    loader.set_max_image_size(100)
    pages = loader.load_pages(str(pdf_source), [3, 1])
    from PIL import Image
    assert [p['page_number'] for p in pages] == [3, 1]
    with Image.open(BytesIO(pages[0]['image'])) as image:
        assert image.format == 'PNG'
        assert max(image.size) <= 100


@pytest.mark.parametrize('config, kwargs', [
    (CamelotConfig, {'read_pdf_kwargs': {'pages': '1'}}),
    (TabulaConfig, {'read_pdf_kwargs': {'output_format': 'dataframe'}}),
    (TabulaConfig, {'lattice': True, 'stream': True}),
])
def test_reserved_options_rejected(config, kwargs):
    with pytest.raises(ValueError):
        config(**kwargs)


@pytest.mark.parametrize('module, loader', [('camelot', DocumentLoaderCamelot), ('tabula', DocumentLoaderTabula)])
def test_unrelated_package_reports_correct_distribution(monkeypatch, module, loader):
    monkeypatch.setitem(sys.modules, module, SimpleNamespace())
    with pytest.raises(ImportError, match=f'{module}-py'):
        loader()


def test_encrypted_pdf_password_reaches_both_backends(monkeypatch, tmp_path):
    path = tmp_path / 'encrypted.pdf'
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((40, 40), 'Protected table')
        pdf.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw='owner', user_pw='reader')
    read = Mock(return_value=[])
    monkeypatch.setitem(sys.modules, 'camelot', SimpleNamespace(read_pdf=read))
    pages = DocumentLoaderCamelot(CamelotConfig(password='reader')).load(str(path))
    assert 'Protected table' in pages[0]['content']
    assert read.call_args.kwargs['password'] == 'reader'

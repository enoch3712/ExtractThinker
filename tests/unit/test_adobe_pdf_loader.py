import json
from io import BytesIO
from zipfile import ZipFile
from unittest.mock import Mock

import pymupdf
import pytest
from extract_thinker import DocumentLoaderAdobePDF, AdobePDFConfig


def archive(elements=None):
    document = {
        'pages': [{'page_number': 0, 'width': 100, 'height': 200},
                  {'page_number': 1, 'width': 100, 'height': 200}],
        'elements': elements if elements is not None else [
            {'Page': 0, 'Text': 'Scanned invoice', 'Bounds': [10, 20, 60, 160]},
            {'Page': 0, 'Path': '//Document/Table', 'filePaths': ['tables/table.csv']},
            {'Page': 0, 'Path': '//Document/Table/P', 'filePaths': ['tables/table.csv']},
        ],
    }
    output = BytesIO()
    with ZipFile(output, 'w') as bundle:
        bundle.writestr('structuredData.json', json.dumps(document))
        bundle.writestr('tables/table.csv', '\ufeffItem,Amount\nBook,\nPen,5\n')
    return output.getvalue()


def test_adobe_maps_text_tables_and_normalized_regions():
    loader = DocumentLoaderAdobePDF(AdobePDFConfig(include_bbox=True), client=object())
    pages = [dict(content='old', tables=[], page_number=n) for n in (1, 2)]
    loader._apply_archive(archive(), pages)
    assert pages[0]['content'] == 'Scanned invoice'
    assert pages[1]['content'] == ''
    assert pages[1]['tables'] == []
    assert pages[0]['tables'] == [[['Item', 'Amount'], ['Book', ''], ['Pen', '5']]]
    box = pages[0]['regions'][0]['bounding_box']
    assert box['page'] == 1
    assert box['x0'] == pytest.approx(.1)
    assert box['y0'] == pytest.approx(.2)
    assert box['y1'] == pytest.approx(.9)


@pytest.mark.parametrize('page_number', [-1, 2, True, None])
def test_invalid_page_identity_fails(page_number):
    loader = DocumentLoaderAdobePDF(client=object())
    with pytest.raises(ValueError, match='invalid source page'):
        loader._apply_archive(archive([{'Page': page_number, 'Text': 'text'}]), [{'content': '', 'tables': []}])


def test_load_preserves_stream_and_uses_cache(monkeypatch):
    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.new_page()
        source = BytesIO(pdf.tobytes())
    source.seek(10)
    loader = DocumentLoaderAdobePDF(client=object())
    extract = Mock(return_value=archive())
    monkeypatch.setattr(loader, '_extract_archive', extract)
    pages = loader.load(source)
    assert len(pages) == 2
    assert pages[0]['content'] == 'Scanned invoice'
    assert source.tell() == 10
    assert loader.load(source) is pages
    extract.assert_called_once_with(source.getvalue())


def test_tables_can_be_disabled():
    loader = DocumentLoaderAdobePDF(AdobePDFConfig(extract_tables=False), client=object())
    pages = [{'content': '', 'tables': []}, {'content': '', 'tables': []}]
    loader._apply_archive(archive(), pages)
    assert pages[0]['tables'] == []


def test_secrets_not_in_config_repr():
    config = AdobePDFConfig(client_id='private-id', client_secret='private-secret')
    assert 'private-id' not in repr(config)
    assert 'private-secret' not in repr(config)

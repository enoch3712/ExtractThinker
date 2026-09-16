"""Page selection works across document loaders (#310)."""
from unittest.mock import Mock
import pytest
from extract_thinker import DocumentLoaderData, DocumentLoaderPyPdf
from pypdf import PdfWriter
from io import BytesIO


def test_selection_preserves_requested_order_and_does_not_mutate_cache():
    pages = [{'content': 'first'}, {'content': 'second'}, {'content': 'third'}]
    loader = DocumentLoaderData()
    loader.load = Mock(return_value=pages)
    selected = loader.load_pages('source', [3, 1])
    assert selected == [{'content': 'third', 'page_number': 3}, {'content': 'first', 'page_number': 1}]
    assert 'page_number' not in pages[0]
    selected[0]['content'] = 'edited'
    assert pages[2]['content'] == 'third'


@pytest.mark.parametrize('pages', [[0], [-1], [True], [1.5], [1, 1]])
def test_invalid_selection_rejected_before_loading(pages):
    loader = DocumentLoaderData()
    loader.load = Mock()
    with pytest.raises(ValueError):
        loader.load_pages('source', pages)
    loader.load.assert_not_called()


def test_selection_out_of_range():
    with pytest.raises(ValueError, match='contains 1 pages'):
        DocumentLoaderData().load_pages('one page', [2])


def test_empty_selection_does_not_load_document():
    loader = DocumentLoaderData()
    loader.load = Mock()
    assert loader.load_pages('source', []) == []
    loader.load.assert_not_called()


def test_page_selection_on_real_pdf_stream():
    writer = PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=100, height=100)
    stream = BytesIO()
    writer.write(stream)
    loader = DocumentLoaderPyPdf()
    assert [p['page_number'] for p in loader.load_pages(stream, [3, 1])] == [3, 1]

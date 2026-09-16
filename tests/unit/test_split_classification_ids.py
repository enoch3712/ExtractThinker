import asyncio
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock

import pytest
from PIL import Image
from extract_thinker import Classification, TextSplitter, ImageSplitter, Process
from extract_thinker.models.doc_group import DocGroup
from extract_thinker.models.split_classification import (
    NumericDocumentGroups, NumericDocumentGroup, NumericPagePair, resolve_classification,
)


def classifications():
    return [Classification(name='Invoice', description='Sales invoice'),
            Classification(name='Invoice', description='Purchase invoice')]


def response(groups):
    return NumericDocumentGroups(reasoning='documents', groupOfDocuments=[
        NumericDocumentGroup(pages=pages, classification=identifier) for pages, identifier in groups])


@pytest.fixture(params=[TextSplitter, ImageSplitter])
def splitter_and_pages(request):
    splitter = request.param('test/model')
    splitter.llm = Mock()
    image = BytesIO()
    Image.new('RGB', (5, 5)).save(image, format='PNG')
    return splitter, [dict(content='invoice', image=image.getvalue()) for _ in range(3)]


def test_eager_numeric_ids_disambiguate_identical_names(splitter_and_pages):
    splitter, pages = splitter_and_pages
    splitter.llm.request.return_value = response([([1, 2], 2), ([3], 1)])
    result = splitter.split_eager_doc_group(pages, classifications())
    assert [(group.pages, group.classification_id) for group in result] == [([1, 2], 2), ([3], 1)]
    assert result[0].classification == 'Invoice'
    assert 'ID 2: Invoice' in str(splitter.llm.request.call_args.kwargs['messages'])


@pytest.mark.parametrize('groups', [([([1], 1)]), [([1, 1, 2, 3], 1)], [([2, 1, 3], 1)], [([1, 2, 3], 8)]])
def test_invalid_eager_results_raise(splitter_and_pages, groups):
    splitter, pages = splitter_and_pages
    splitter.llm.request.return_value = response(groups)
    with pytest.raises(ValueError):
        splitter.split_eager_doc_group(pages, classifications())


def test_provider_failure_never_selects_first_classification(splitter_and_pages):
    splitter, pages = splitter_and_pages
    splitter.llm.request.side_effect = RuntimeError('provider unavailable')
    with pytest.raises(RuntimeError, match='provider unavailable'):
        splitter.split_eager_doc_group(pages, classifications())
    with pytest.raises(RuntimeError, match='provider unavailable'):
        splitter.split_lazy_doc_group(pages, classifications())


def test_lazy_ids_and_consistency(splitter_and_pages):
    splitter, pages = splitter_and_pages
    splitter.llm.request.side_effect = [
        NumericPagePair(belongs_to_same_document=True, classification_page1=2, classification_page2=2),
        NumericPagePair(belongs_to_same_document=False, classification_page1=2, classification_page2=1),
    ]
    result = splitter.split_lazy_doc_group(pages, classifications()).doc_groups
    assert [(g.pages, g.classification_id) for g in result] == [([1, 2], 2), ([3], 1)]
    splitter.llm.request.side_effect = [
        NumericPagePair(belongs_to_same_document=True, classification_page1=2, classification_page2=2),
        NumericPagePair(belongs_to_same_document=False, classification_page1=1, classification_page2=1),
    ]
    with pytest.raises(ValueError, match='Conflicting classifications'):
        splitter.split_lazy_doc_group(pages, classifications())


def test_empty_and_single_page_lazy(splitter_and_pages):
    splitter, pages = splitter_and_pages
    assert splitter.split_lazy_doc_group([], classifications()).doc_groups == []
    splitter.llm.request.assert_not_called()
    splitter.llm.request.return_value = response([([1], 2)])
    assert splitter.split_lazy_doc_group(pages[:1], classifications()).doc_groups[0].classification_id == 2


def test_legacy_names_must_be_unambiguous():
    with pytest.raises(ValueError, match='ambiguous'):
        resolve_classification(classifications(), name='Invoice')
    assert resolve_classification(classifications()[:1], name='Invoice').description == 'Sales invoice'


def test_process_uses_id_even_when_names_identical():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    first, second = Mock(), Mock()
    first.extract_async = AsyncMock(return_value='wrong')
    second.extract_async = AsyncMock(return_value='purchase')
    process = Process()
    process.split_classifications = classifications()
    process.split_classifications[0].extractor = first
    process.split_classifications[1].extractor = second
    process.doc_groups = [DocGroup([1], 'Invoice', 2)]
    process.document_loader = SimpleNamespace(load=lambda path: [{'content': 'purchase'}])
    process.file_path = 'document.pdf'
    try:
        assert process.extract() == ['purchase']
        first.extract_async.assert_not_called()
        second.extract_async.assert_awaited_once()
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def test_image_example_path_is_retained_with_correct_mime(tmp_path):
    path = tmp_path / 'example.png'
    Image.new('RGB', (5, 5)).save(path)
    splitter = ImageSplitter('test/model')
    splitter.llm = Mock()
    splitter.llm.request.return_value = response([([1], 1)])
    labels = classifications()[:1]
    labels[0].set_image(str(path))
    splitter.split_eager_doc_group([{'image': path.read_bytes()}], labels)
    content = splitter.llm.request.call_args.kwargs['messages'][0]['content']
    images = [item['image_url']['url'] for item in content if item['type'] == 'image_url']
    assert len(images) == 2
    assert all(url.startswith('data:image/png;base64,') for url in images)

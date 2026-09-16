import pytest
from pydantic import ValidationError
from extract_thinker import BoundingBox, Contract, DocumentRegion, Signature


def test_evidence_types_embed_in_contract_and_round_trip():
    class SignedDocument(Contract):
        signature: Signature
    box = BoundingBox(page=2, x0=0.1, y0=0.2, x1=0.8, y1=0.4)
    result = SignedDocument(signature=Signature(present=True, bounding_box=box))
    assert SignedDocument.model_validate_json(result.model_dump_json()) == result
    assert DocumentRegion(text='signature', bounding_box=box).confidence is None


@pytest.mark.parametrize('change', [
    {'page': 0}, {'x0': -0.1}, {'y1': 1.1}, {'x0': 0.9, 'x1': 0.1},
    {'y0': 0.9, 'y1': 0.1}, {'x0': float('nan')},
])
def test_invalid_coordinates_rejected(change):
    data = dict(page=1, x0=0, y0=0, x1=1, y1=1)
    data.update(change)
    with pytest.raises(ValidationError):
        BoundingBox(**data)


def test_extractor_preserves_source_regions_and_tables_in_model_input():
    from extract_thinker import Extractor
    pages = [
        {'content': 'page one'},
        {'content': 'signed here', 'page_number': 3,
         'regions': [{'text': 'signature', 'bounding_box': dict(page=3, x0=0, y0=0, x1=1, y1=1)}],
         'tables': [[['Item', 'Price']]]},
    ]
    extractor = Extractor()
    mapped = extractor._map_to_universal_format(pages)
    assert mapped['metadata']['pages'][1]['page_number'] == 3
    assert mapped['metadata']['pages'][1]['regions'] == pages[1]['regions']
    assert mapped['metadata']['pages'][1]['tables'] == pages[1]['tables']
    assert 'bounding_box' in extractor._convert_content_to_string(mapped)


def test_preloaded_pages_keep_coordinates_through_full_extraction():
    from unittest.mock import Mock
    from extract_thinker import Extractor, DocumentLoaderData
    pages = [{'content': 'signed here', 'page_number': 3,
              'regions': [{'text': 'signature', 'bounding_box': dict(page=3, x0=0, y0=0, x1=1, y1=1)}]}]
    extractor = Extractor(DocumentLoaderData())
    extractor.llm = Mock()
    extractor.extract(pages, Signature)
    messages = extractor.llm.request.call_args.args[0]
    assert 'bounding_box' in str(messages)
    assert 'page_number: 3' in str(messages)
    assert 'images' not in pages[0]

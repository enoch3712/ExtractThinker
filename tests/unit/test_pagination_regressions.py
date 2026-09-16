"""Long-document extraction must preserve page association and contract semantics."""
from concurrent.futures import Future
from unittest.mock import Mock
from typing import Optional
import pytest
from pydantic import BaseModel, Field, ValidationError, model_validator
from extract_thinker import Extractor, DocumentLoaderTxt, CompletionStrategy
from extract_thinker.pagination_handler import PaginationHandler
from extract_thinker.utils import make_all_fields_optional


class Invoice(BaseModel):
    number: str = Field(description='Invoice identifier, not customer identifier')
    note: Optional[str] = 'default note'


def test_partial_model_preserves_field_metadata_and_defers_model_validation():
    class Enriched(BaseModel):
        number: str = Field(alias='invoiceNumber', description='Invoice identifier')
        customer: str
        @model_validator(mode='after')
        def check_invoice(self):
            if self.customer != 'ACME':
                raise ValueError('invalid customer')
            return self
    partial = make_all_fields_optional(Enriched)
    result = partial(invoiceNumber='INV-123')
    assert result.customer is None
    assert partial.model_fields['number'].description == 'Invoice identifier'
    assert partial.model_fields['number'].alias == 'invoiceNumber'
    with pytest.raises(ValidationError, match='invalid customer'):
        Enriched.model_validate(dict(number='INV-123', customer='other'), by_name=True)


def test_missing_required_field_is_not_replaced_with_empty_string():
    partial = make_all_fields_optional(Invoice)()
    with pytest.raises(ValidationError):
        PaginationHandler(Mock())._merge_results([partial], Invoice, [({}, partial)])


def test_absent_optional_field_uses_contract_default():
    partial = make_all_fields_optional(Invoice)(number='INV-123')
    result = PaginationHandler(Mock())._merge_results([partial], Invoice, [({}, partial)])
    assert result.note == 'default note'


def test_aliases_validate_after_merge():
    class Aliased(BaseModel):
        number: str = Field(alias='invoiceNumber')
    partial = make_all_fields_optional(Aliased)(invoiceNumber='INV-123')
    result = PaginationHandler(Mock())._merge_results([partial], Aliased, [({}, partial)])
    assert result.number == 'INV-123'


def test_unresolved_conflict_is_not_arbitrarily_chosen():
    with pytest.raises(ValueError, match='Unresolved extraction conflict'):
        PaginationHandler(Mock())._clean_merged_dict(
            {'number': {'_conflict': True, 'candidates': ['one', 'two']}}, Invoice)


def test_page_errors_fail_the_document():
    handler = PaginationHandler(Mock())
    handler._process_page = Mock(side_effect=ValueError('provider failure'))
    with pytest.raises(ValueError, match='no partial document returned'):
        handler.handle([{'content': 'page'}], Invoice)


def test_out_of_order_completions_keep_page_result_associations(monkeypatch):
    # Use already-resolved futures and reverse completion order deterministically.
    class Executor:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, function, *args):
            future = Future()
            future.set_result(function(*args))
            return future
    monkeypatch.setattr('extract_thinker.pagination_handler.ThreadPoolExecutor', Executor)
    monkeypatch.setattr('extract_thinker.pagination_handler.as_completed', lambda fs: reversed(list(fs)))
    handler = PaginationHandler(Mock())
    handler._process_page = Mock(side_effect=[Invoice(number='first'), Invoice(number='second')])
    handler._merge_results = Mock(return_value='merged')
    pages = [{'content': 'first'}, {'content': 'second'}]
    assert handler.handle(pages, Invoice) == 'merged'
    associations = handler._merge_results.call_args.args[2]
    assert [(page['content'], result.number) for page, result in associations] == [('first', 'first'), ('second', 'second')]


def test_strategy_loads_file_lists_before_extraction(tmp_path, monkeypatch):
    paths = []
    for index in range(2):
        path = tmp_path / f'{index}.txt'
        path.write_text(f'Invoice page {index}')
        paths.append(str(path))
    extractor = Extractor(DocumentLoaderTxt())
    extractor.llm = Mock()
    handle = Mock(return_value=Invoice(number='INV-123'))
    monkeypatch.setattr('extract_thinker.extractor.PaginationHandler.handle', handle)
    extractor.extract(paths, Invoice, completion_strategy=CompletionStrategy.PAGINATE)
    loaded = handle.call_args.args[0]
    assert [p['content'] for p in loaded] == ['Invoice page 0', 'Invoice page 1']


def test_seven_page_vision_document_uses_one_page_per_request():
    from typing import List
    from io import BytesIO
    from PIL import Image
    from extract_thinker import DocumentLoaderData

    class Packet(BaseModel):
        number: str
        rows: List[int]

    image = BytesIO()
    Image.new('RGB', (2, 2), 'white').save(image, format='PNG')
    pages = [{'content': f'page {index}', 'image': image.getvalue()} for index in range(7)]
    requests = []
    def request(messages, response_model):
        payload = messages[-1]['content']
        images = [part for part in payload if part['type'] == 'image_url']
        assert len(images) == 1  # A provider accepting only one page per call.
        index = int(payload[0]['text'].split()[-1])
        requests.append(index)
        return response_model(number='INV-123', rows=[index])

    extractor = Extractor(DocumentLoaderData())
    extractor.llm = Mock()
    extractor.llm.request.side_effect = request
    result = extractor.extract(pages, Packet, vision=True,
                               completion_strategy=CompletionStrategy.PAGINATE)
    assert result.number == 'INV-123'
    assert result.rows == list(range(7))
    assert sorted(requests) == list(range(7))

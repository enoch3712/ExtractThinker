from io import BytesIO
from threading import Barrier
from typing import Annotated
from unittest.mock import Mock
from pydantic import Field, model_validator, field_validator, AfterValidator
from PIL import Image
import pytest
from extract_thinker import Contract, Extractor, LLM, FieldExtraction
from extract_thinker.document_loader.document_loader import DocumentLoader


class Loader(DocumentLoader):
    def __init__(self):
        super().__init__()
        self.calls = 0
    def can_handle(self, source):
        return True
    def load(self, source):
        self.calls += 1
        return [{'content': 'Invoice 12, tax 2'}]


def test_fields_run_concurrently_and_final_validators_run_once():
    validations = []
    class Invoice(Contract):
        subtotal: int
        tax: int
        @model_validator(mode='after')
        def check_total(self):
            validations.append((self.subtotal, self.tax))
            assert self.subtotal > self.tax
            return self
    barrier = Barrier(2)
    llm = LLM('test/model')
    def request(messages, response_model):
        barrier.wait(timeout=5)
        values = {'subtotal': 12, 'tax': 2}
        return response_model.model_validate({name: values[name] for name in response_model.model_fields})
    llm.request = Mock(side_effect=request)
    loader = Loader()
    extractor = Extractor(loader, llm)
    result = extractor.extract_fields('file', Invoice, max_workers=2)
    assert result.subtotal == 12 and result.tax == 2
    assert validations == [(12, 2)]
    assert loader.calls == 1
    assert llm.page_count is None
    assert loader.vision_mode is False


def test_annotations_route_models_and_trigger_automatic_field_extraction():
    small, large = LLM('test/small'), LLM('test/large')
    small.request = Mock(side_effect=lambda messages, model: model(summary='long text'))
    large.request = Mock(side_effect=lambda messages, model: model(amount=12))
    class Invoice(Contract):
        summary: Annotated[str, FieldExtraction(model=small, instructions='transcribe exactly')]
        amount: Annotated[int, FieldExtraction(model=large)]
    result = Extractor(Loader(), large).extract('file', Invoice)
    assert result.summary == 'long text'
    assert result.amount == 12
    assert 'transcribe exactly' in str(small.request.call_args)
    small.request.assert_called_once()
    large.request.assert_called_once()


def test_grouping_preserves_aliases_constraints_and_defers_field_validators():
    validations = []
    policy = FieldExtraction(group='totals')
    class Invoice(Contract):
        amount: Annotated[int, policy] = Field(alias='totalAmount', gt=0)
        tax: Annotated[int, policy]
        @field_validator('amount')
        @classmethod
        def enrich(cls, value):
            validations.append(value)
            return value + 1
    llm = LLM('test/model')
    def request(messages, model):
        assert model.model_json_schema()['properties']['totalAmount']['exclusiveMinimum'] == 0
        return model(totalAmount=12, tax=2)
    llm.request = Mock(side_effect=request)
    result = Extractor(Loader(), llm).extract('source', Invoice)
    assert result.amount == 13
    assert validations == [12]
    llm.request.assert_called_once()


def test_annotated_validator_and_default_factory_apply_only_at_final_merge():
    defaults = []
    def factory():
        defaults.append(True)
        return 'generated'
    class Data(Contract):
        value: Annotated[int, FieldExtraction(), AfterValidator(lambda value: value + 1)]
        label: str = Field(default_factory=factory)
    llm = LLM('test/model')
    llm.request = Mock(side_effect=lambda messages, model: model(value=2) if 'value' in model.model_fields else model())
    result = Extractor(Loader(), llm).extract('source', Data)
    assert result.value == 3
    assert result.label == 'generated'
    assert defaults == [True]


def test_failure_never_returns_partial_contract():
    class Data(Contract):
        first: str
        second: str
    llm = LLM('test/model')
    def request(messages, model):
        if 'second' in model.model_fields:
            raise RuntimeError('provider failure')
        return model(first='first')
    llm.request = Mock(side_effect=request)
    with pytest.raises(ValueError, match='second.*provider failure'):
        Extractor(Loader(), llm).extract_fields('source', Data)


def test_inconsistent_group_policies_rejected_before_loading():
    class Data(Contract):
        first: Annotated[str, FieldExtraction(group='same', vision=True)]
        second: Annotated[str, FieldExtraction(group='same', vision=False)]
    loader = Loader()
    with pytest.raises(ValueError, match='same extraction policy'):
        Extractor(loader, LLM('test/model')).extract('source', Data)
    assert loader.calls == 0


def test_field_vision_only_sends_images_to_vision_group():
    class Data(Contract):
        text: Annotated[str, FieldExtraction(vision=False)]
        chart: Annotated[str, FieldExtraction(vision=True)]
    image = BytesIO()
    Image.new('RGB', (5, 5)).save(image, format='PNG')
    pages = [{'content': 'Chart text', 'image': image.getvalue()}]
    llm = LLM('test/model')
    calls = {}
    def request(messages, model):
        name = next(iter(model.model_fields))
        calls[name] = str(messages)
        return model(**{name: 'result'})
    llm.request = Mock(side_effect=request)
    result = Extractor(llm=llm).extract(pages, Data)
    assert result.chart == 'result'
    assert 'image_url' in calls['chart']
    assert 'image_url' not in calls['text']
    assert pages[0]['image'] == image.getvalue()


def test_annotated_models_work_with_pagination():
    from extract_thinker import CompletionStrategy
    llm = LLM('test/model')
    class Data(Contract):
        value: Annotated[int, FieldExtraction(model=llm)]
    llm.request = Mock(side_effect=lambda messages, model: model(value=7))
    result = Extractor(llm=llm).extract(
        [{'content': 'first'}, {'content': 'second'}], Data,
        completion_strategy=CompletionStrategy.PAGINATE,
    )
    assert result.value == 7
    assert llm.request.call_count == 2


def test_cross_field_validation_failure_is_not_suppressed():
    class Data(Contract):
        first: int
        second: int
        @model_validator(mode='after')
        def check_order(self):
            if self.first >= self.second:
                raise ValueError('fields are inconsistent')
            return self
    llm = LLM('test/model')
    llm.request = Mock(side_effect=lambda messages, model: model(**{name: 1 for name in model.model_fields}))
    with pytest.raises(ValueError, match='fields are inconsistent'):
        Extractor(Loader(), llm).extract_fields('source', Data)


def test_loading_failure_restores_vision_mode():
    class Data(Contract):
        field: Annotated[str, FieldExtraction(vision=True)]
    loader = Loader()
    loader.load = Mock(side_effect=RuntimeError('loader failed'))
    with pytest.raises(RuntimeError, match='loader failed'):
        Extractor(loader, LLM('test/model')).extract('source', Data)
    assert loader.vision_mode is False

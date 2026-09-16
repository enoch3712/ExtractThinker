from unittest.mock import Mock
from typing import List
import pytest
from pydantic import BaseModel
from extract_thinker import ComplexityRouter, ModelRoute, LLM, Extractor


class SimpleContract(BaseModel):
    name: str


class Item(BaseModel):
    name: str
    amount: float


class NestedContract(BaseModel):
    items: List[Item]
    total: float


def providers():
    small, large = LLM('test/small'), LLM('test/large')
    small.request = Mock(return_value='small result')
    large.request = Mock(return_value='large result')
    small.raw_completion = Mock(return_value='small raw')
    large.raw_completion = Mock(return_value='large raw')
    return small, large


def test_routes_text_length_and_contract_complexity():
    small, large = providers()
    router = ComplexityRouter([ModelRoute(small, max_score=2), ModelRoute(large)])
    assert router.request([{'role': 'user', 'content': 'short'}], SimpleContract) == 'small result'
    assert router.request([{'role': 'user', 'content': 'x' * 8000}], SimpleContract) == 'large result'
    assert router.request([{'role': 'user', 'content': 'short'}], NestedContract) == 'large result'
    assert router.last_decision.complexity.field_count == 4
    assert router.last_decision.complexity.schema_depth == 2
    assert router.last_decision.model == 'test/large'


def test_vision_never_routes_to_text_only_model_or_counts_base64_as_text():
    small, large = providers()
    router = ComplexityRouter([ModelRoute(small, max_score=100), ModelRoute(large, supports_vision=True)])
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'read'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + 'a' * 10000}}]}]
    assert router.request(messages, SimpleContract) == 'large result'
    assert router.last_decision.complexity.text_characters == 4
    assert router.last_decision.complexity.image_count == 1
    small.request.assert_not_called()


def test_no_capable_route_fails_before_provider_call():
    small, _ = providers()
    router = ComplexityRouter([ModelRoute(small)])
    with pytest.raises(ValueError, match='No configured route'):
        router.request([{'content': [{'type': 'image_url', 'image_url': {'url': 'image'}}]}], SimpleContract)
    small.request.assert_not_called()


def test_custom_scorer_page_count_and_raw_calls():
    small, large = providers()
    router = ComplexityRouter([ModelRoute(small, max_score=2), ModelRoute(large)], scorer=lambda info: info.page_count)
    router.set_page_count(3)
    assert router.raw_completion([{'content': 'text'}]) == 'large raw'
    assert large.page_count == 3
    assert router.last_decision.score == 3


def test_provider_failure_is_not_hidden_or_retried_on_another_route():
    small, large = providers()
    router = ComplexityRouter([ModelRoute(small, max_score=10), ModelRoute(large)])
    small.request.side_effect = RuntimeError('provider failed')
    with pytest.raises(RuntimeError, match='provider failed'):
        router.request([{'content': 'text'}], SimpleContract)
    large.request.assert_not_called()


@pytest.mark.parametrize('routes', [[], [ModelRoute('a'), ModelRoute('b')],
    [ModelRoute('a', -1)], [ModelRoute('a', float('nan'))],
    [ModelRoute('a', 3), ModelRoute('b', 2)], [ModelRoute('')]])
def test_invalid_configuration_rejected(routes):
    with pytest.raises(ValueError):
        ComplexityRouter(routes)


def test_invalid_custom_score_rejected():
    router = ComplexityRouter([ModelRoute('test/model')], scorer=lambda info: float('inf'))
    with pytest.raises(ValueError, match='finite nonnegative'):
        router.request([{'content': 'text'}], SimpleContract)


def test_batch_processing_cannot_bypass_routing():
    router = ComplexityRouter([ModelRoute('gpt-4o')])
    extractor = Extractor(llm=router)
    assert not extractor.can_handle_batch()


def test_completion_metadata_is_forwarded_and_cleared_on_failure():
    small, _ = providers()
    marker = object()
    small.last_completion = marker
    router = ComplexityRouter([ModelRoute(small)])
    router.request([{'content': 'text'}], SimpleContract)
    assert router.last_completion is marker
    small.request.side_effect = RuntimeError('failed')
    with pytest.raises(RuntimeError):
        router.request([{'content': 'text'}], SimpleContract)
    assert router.last_completion is None

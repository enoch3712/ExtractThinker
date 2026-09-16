"""Offline adapter regressions for #352 and #356."""
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
from extract_thinker import LLM


@pytest.mark.parametrize('limit', [1, 4000, 16000])
def test_explicit_token_limit_overrides_completion_default(limit):
    llm = LLM('test-model', token_limit=limit)
    llm.client = Mock()
    llm.request([{'role': 'user', 'content': 'test'}])
    assert llm.client.chat.completions.create.call_args.kwargs['max_completion_tokens'] == limit


def test_page_and_reasoning_budgets_respect_small_limit():
    llm = LLM('test-model', token_limit=4000)
    llm.set_page_count(100)
    assert llm.thinking_token_limit == 4000
    assert llm.thinking_budget < 4000


@pytest.mark.parametrize('limit', [0, -1, True, 1.5])
def test_invalid_token_limit(limit):
    with pytest.raises(ValueError, match='positive integer'):
        LLM('test-model', token_limit=limit)


@pytest.mark.parametrize('count', [0, -1, True, 1.5])
def test_invalid_page_count(count):
    with pytest.raises(ValueError, match='positive integer'):
        LLM('test-model').set_page_count(count)


def test_provider_options_and_raw_metadata():
    options = {'logprobs': True, 'top_logprobs': 3}
    llm = LLM('test-model', completion_kwargs=options)
    options['top_logprobs'] = 99
    llm.client = Mock()
    raw = NS(choices=[NS(logprobs='probabilities')])
    llm.client.chat.completions.create.return_value = NS(_raw_response=raw)
    llm.request([{'role': 'user', 'content': 'test'}])
    params = llm.client.chat.completions.create.call_args.kwargs
    assert params['logprobs'] is True
    assert params['top_logprobs'] == 3
    assert llm.last_completion is raw


@pytest.mark.parametrize('key', ['messages', 'model', 'response_model', 'stream', 'max_tokens', 'max_completion_tokens'])
def test_provider_options_cannot_override_contract(key):
    with pytest.raises(ValueError, match='cannot override'):
        LLM('test-model', completion_kwargs={key: 'unsafe override'})


def test_provider_options_forwarded_to_router_and_raw(monkeypatch):
    llm = LLM('test-model', token_limit=12000, completion_kwargs={'logprobs': True})
    raw = NS(choices=[NS(message=NS(content='{}'))])
    completion = Mock(return_value=raw)
    monkeypatch.setattr('extract_thinker.llm.litellm.completion', completion)
    assert llm.raw_completion([]) == '{}'
    assert llm.last_completion is raw
    assert completion.call_args.kwargs['max_completion_tokens'] == 12000
    assert completion.call_args.kwargs['logprobs'] is True
    llm.router = Mock()
    llm._request_with_router([], None)
    assert llm.router.completion.call_args.kwargs['logprobs'] is True

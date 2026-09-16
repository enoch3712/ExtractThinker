import json
from unittest.mock import Mock
import pytest
from litellm import ModelResponse
from pydantic import Field
from extract_thinker import Contract, DocumentLoaderData, Extractor, LLM
from extract_thinker.llm_engine import LLMEngine


class Notification(Contract):
    notification_number: int = Field(description='Notification number')


def response(content):
    return ModelResponse(model='ollama_chat/test', choices=[{
        'index': 0, 'finish_reason': 'stop',
        'message': {'role': 'assistant', 'content': json.dumps(content)},
    }])


def test_seven_pages_use_native_schema_and_context_options(monkeypatch):
    completion = Mock(return_value=response({'notification_number': 1234}))
    monkeypatch.setattr('extract_thinker.llm.litellm.completion', completion)
    llm = LLM('ollama_chat/test', structured_output=True, token_limit=1000,
              completion_kwargs={'api_base': 'http://localhost:11434', 'num_ctx': 32768})
    pages = [{'content': f'Page {i}: notification 1234. ' * 200, 'page_number': i} for i in range(1, 8)]
    result = Extractor(DocumentLoaderData(), llm).extract(pages, Notification)
    assert result.notification_number == 1234
    params = completion.call_args.kwargs
    assert params['response_format']['type'] == 'json_schema'
    schema = params['response_format']['json_schema']['schema']
    assert schema['required'] == ['notification_number']
    from litellm.llms.ollama.chat.transformation import OllamaChatConfig
    mapped = OllamaChatConfig().map_openai_params(
        {'response_format': params['response_format']}, {}, 'test', False)
    assert mapped['format'] == schema
    assert params['num_ctx'] == 32768
    assert params['max_completion_tokens'] == 1000
    prompt = json.dumps(params['messages'])
    assert all(f'Page {i}:' in prompt for i in range(1, 8))


def test_wrong_contract_is_rejected_even_in_native_mode(monkeypatch):
    completion = Mock(return_value=response({'documents': [{'summary': 'wrong shape'}]}))
    monkeypatch.setattr('extract_thinker.llm.litellm.completion', completion)
    llm = LLM('ollama_chat/test', structured_output=True)
    with pytest.raises(Exception, match='notification_number'):
        llm.request([{'role': 'user', 'content': 'notification'}], Notification)


def test_structured_mode_rejects_bypasses():
    with pytest.raises(ValueError, match='response_format'):
        LLM('test', structured_output=True, completion_kwargs={'response_format': {}})
    with pytest.raises(ValueError, match='default LiteLLM'):
        LLM('test', structured_output=True, backend=LLMEngine.PYDANTIC_AI)
    llm = LLM('test', structured_output=True)
    with pytest.raises(ValueError, match='Dynamic'):
        llm.set_dynamic(True)
    with pytest.raises(ValueError, match='router'):
        llm.load_router(Mock())

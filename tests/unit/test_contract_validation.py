"""Exercise the real Instructor adapter with an offline completion transport."""
from unittest.mock import Mock
from typing import Optional
from litellm import ModelResponse
from pydantic import model_validator
from extract_thinker import Contract, Extractor, DocumentLoaderData, LLM


class Invoice(Contract):
    tax_id: str
    internal_id: Optional[int] = None

    @model_validator(mode='after')
    def attach_internal_id(self):
        # Stand-in for an application-owned database lookup (#46).
        self.internal_id = {'ACME': 42}[self.tax_id]
        return self


def test_contract_post_validation_runs_after_provider_call(monkeypatch):
    completion = Mock(return_value=ModelResponse(
        model='test-model',
        choices=[{'index': 0, 'finish_reason': 'stop',
                  'message': {'role': 'assistant', 'content': '{"tax_id":"ACME"}'}}],
    ))
    monkeypatch.setattr('extract_thinker.llm.litellm.completion', completion)
    llm = LLM('test-model')
    extractor = Extractor(DocumentLoaderData())
    extractor.load_llm(llm)
    result = extractor.extract('Invoice for ACME', Invoice)
    assert result.tax_id == 'ACME'
    assert result.internal_id == 42
    assert llm.last_completion is not None
    assert completion.call_count == 1

"""Output continuation regressions for #326 and #351."""
from unittest.mock import Mock
import pytest
from pydantic import BaseModel
from extract_thinker.concatenation_handler import ConcatenationHandler


class Result(BaseModel):
    name: str
    count: int


@pytest.mark.parametrize('fragments', [
    ['{"name":"Hello ', 'world","count":', '12}'],
    ['```json\n{"name":"Hello world","count":12}\n```'],
    ['{"name":"Hello world",', '"count":12}'],
])
def test_continuation_accepts_scalar_fragments_and_preserves_spaces(fragments):
    llm = Mock()
    llm.raw_completion.side_effect = fragments
    result = ConcatenationHandler(llm).handle('input document', Result)
    assert result == Result(name='Hello world', count=12)
    assert llm.raw_completion.call_count == len(fragments)


def test_invalid_schema_requests_replacement_not_concatenation():
    llm = Mock()
    llm.raw_completion.side_effect = ['{"unexpected":"value"}', '{"name":"correct","count":1}']
    assert ConcatenationHandler(llm).handle('input', Result).name == 'correct'
    assert 'complete JSON' in llm.raw_completion.call_args.args[0][-1]['content']


def test_provider_error_is_not_hidden_by_continuation_retry():
    llm = Mock()
    llm.raw_completion.side_effect = ValueError('provider context limit')
    with pytest.raises(ValueError, match='provider context limit'):
        ConcatenationHandler(llm).handle('input', Result)
    assert llm.raw_completion.call_count == 1


def test_retries_are_bounded_and_explain_input_vs_output_limits():
    llm = Mock()
    llm.raw_completion.return_value = ''
    with pytest.raises(ValueError, match='PAGINATE'):
        ConcatenationHandler(llm).handle('input', Result)
    assert llm.raw_completion.call_count == 4

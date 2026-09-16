from unittest.mock import Mock
import pytest
from extract_thinker import MarkdownSplitter, Classification
from extract_thinker.models.split_classification import NumericDocumentGroups as DocGroupsEager, NumericDocumentGroup as DocGroup


def test_sections_preserve_text_and_ignore_fenced_headings():
    source = 'Intro\n\n# First\nbody\n```python\n# not a heading\n```\n\n## Second\nlast\n'
    result = MarkdownSplitter().split_sections(source)
    assert [section['heading'] for section in result] == [None, 'First', 'Second']
    assert ''.join(section['content'] for section in result) == source
    assert result[1]['start_line'] == 3
    assert result[2]['start_line'] == 9
    assert result[-1]['end_line'] == 10


def test_setext_heading_and_configurable_depth():
    source = 'Top\n===\n\n## Nested\nbody\n'
    assert len(MarkdownSplitter(heading_level=1).split_sections(source)) == 1
    assert len(MarkdownSplitter(heading_level=2).split_sections(source)) == 2


def test_empty_and_heading_free_markdown():
    assert MarkdownSplitter().split_sections('') == []
    result = MarkdownSplitter().split_sections('Only text')
    assert result[0]['content'] == 'Only text'
    assert result[0]['heading'] is None


def test_page_strategy_prefers_markdown_without_mutating_input():
    splitter = MarkdownSplitter()
    splitter.llm = Mock()
    splitter.llm.request.return_value = DocGroupsEager(reasoning='same invoice', groupOfDocuments=[
        DocGroup(pages=[1, 2], classification=1)])
    pages = [{'content': 'plain text', 'markdown': '# Invoice\n| Item | Price |'},
             {'content': 'plain continuation', 'markdown': '## Continued'}]
    groups = splitter.split_eager_doc_group(pages, [Classification(name='Invoice', description='Invoice')])
    assert groups[0].pages == [1, 2]
    prompt = str(splitter.llm.request.call_args.kwargs['messages'])
    assert '# Invoice' in prompt and '| Item | Price |' in prompt
    assert 'plain text' not in prompt
    assert pages[0]['content'] == 'plain text'


def test_semantic_strategy_requires_model():
    with pytest.raises(ValueError, match='model is required'):
        MarkdownSplitter().split_eager_doc_group([], [])

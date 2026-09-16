import re
from unittest.mock import Mock
import pytest
from extract_thinker import EntityMasker, DocumentLoaderMasked, Extractor, Contract
from extract_thinker.document_loader.document_loader import DocumentLoader


class Loader(DocumentLoader):
    def __init__(self, pages):
        super().__init__()
        self.pages = pages
    def can_handle(self, source):
        return True
    def load(self, source):
        return self.pages


def test_long_names_repeated_entities_and_email_restore_exactly():
    source = 'Alice Smith and Alice met Ann. Anna emailed alice@example.com. Alice Smith replied.'
    masked = EntityMasker(entities={'PERSON': ['Alice', 'Alice Smith', 'Ann']}).mask(source)
    assert 'Alice' not in masked.content
    assert 'alice@example.com' not in masked.content
    assert 'Anna' in masked.content
    assert len(masked.mapping) == 4
    assert masked.restore(masked.content) == source
    assert 'Alice Smith' not in repr(masked)


def test_nested_metadata_masked_without_mutating_input():
    source = [{'content': 'alice@example.com', 'page_number': 4,
               'tables': [[['Email', 'alice@example.com']]],
               'regions': [{'text': 'Alice', 'bounding_box': {'page': 4, 'x0': .1}}]}]
    masked = EntityMasker(entities={'PERSON': ['Alice']}).mask(source)
    assert 'alice@example.com' not in str(masked.content)
    assert 'Alice' not in str(masked.content)
    assert masked.content[0]['page_number'] == 4
    assert source[0]['content'] == 'alice@example.com'
    assert masked.restore(masked.content) == source


def test_custom_patterns_and_independent_namespaces():
    masker = EntityMasker(patterns={'CUSTOMER_ID': r'CUST-\d+'}, detect_emails=False)
    first = masker.mask('CUST-123')
    second = masker.mask('CUST-123')
    assert first.content != second.content
    assert first.restore(first.content) == 'CUST-123'
    assert first.restore(second.content) == second.content


def test_shared_session_handles_multiple_files():
    session = EntityMasker(entities={'PERSON': ['Alice']}).session()
    first = session.mask('Alice')
    second = session.mask('Alice and bob@example.com')
    assert first.content in second.content
    assert session.restore({'first': first.content, 'second': second.content}) == {
        'first': 'Alice', 'second': 'Alice and bob@example.com'}


def test_extractor_prompt_is_masked_and_result_can_be_restored():
    class Contact(Contract):
        email: str
    pages = [{'content': 'Contact Alice at alice@example.com', 'image': b'private pixels'}]
    loader = DocumentLoaderMasked(Loader(pages), EntityMasker(entities={'PERSON': ['Alice']}))
    extractor = Extractor(loader)
    extractor.llm = Mock()
    def request(messages, model):
        prompt = str(messages)
        assert 'Alice' not in prompt
        assert 'alice@example.com' not in prompt
        assert 'private pixels' not in prompt
        token = re.search(r'__ET_[a-f0-9]+_EMAIL_\d+__', prompt).group()
        return model(email=token)
    extractor.llm.request.side_effect = request
    result = extractor.extract('document', Contact)
    assert loader.restore(result.model_dump()) == {'email': 'alice@example.com'}
    assert pages[0]['image'] == b'private pixels'


def test_wrapper_refuses_unredacted_vision_and_restores_loader_state():
    original = Loader([{'content': 'email@example.com', 'images': [b'image']}])
    original.set_vision_mode(True)
    loader = DocumentLoaderMasked(original, EntityMasker())
    with pytest.raises(ValueError, match='does not redact image pixels'):
        loader.set_vision_mode(True)
    pages = loader.load('source')
    assert 'images' not in pages[0]
    assert original.vision_mode is True


def test_reset_drops_old_mapping():
    loader = DocumentLoaderMasked(Loader([{'content': 'alice@example.com'}]), EntityMasker())
    old = loader.load('source')[0]['content']
    assert loader.restore(old) == 'alice@example.com'
    loader.reset()
    assert loader.restore(old) == old
    assert loader.load('source')[0]['content'] != old


@pytest.mark.parametrize('kwargs', [
    {'entities': {'invalid label': ['Alice']}},
    {'entities': {'PERSON': 'Alice'}},
    {'entities': {'PERSON': ['']}},
    {'patterns': {'EMPTY': 'a*'}},
])
def test_invalid_mask_rules_fail_early(kwargs):
    with pytest.raises(ValueError):
        EntityMasker(**kwargs)


def test_zero_width_match_fails_instead_of_inserting_fake_entities():
    with pytest.raises(ValueError, match='empty spans'):
        EntityMasker(patterns={'BAD': r'(?=abc)'}).mask('abc')

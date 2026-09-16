from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
from extract_thinker import SQLitePageRetriever, DocumentLoaderRAG, Extractor, Contract
from extract_thinker.document_loader.document_loader import DocumentLoader


class PagesLoader(DocumentLoader):
    def __init__(self, pages):
        super().__init__()
        self.pages = pages
    def can_handle(self, source):
        return True
    def load(self, source):
        return self.pages


def test_persistence_replacement_and_document_isolation(tmp_path):
    database = str(tmp_path / 'pages.sqlite')
    with SQLitePageRetriever(database) as retriever:
        retriever.index('first', [{'content': 'invoice'}, {'content': 'payment due'}])
        retriever.index('second', [{'content': 'payment secret'}])
        assert retriever.search('first', 'payment')[0].page_number == 2
    with SQLitePageRetriever(database) as retriever:
        assert retriever.search('first', 'payment')[0].page_number == 2
        retriever.index('first', [{'content': 'replacement'}])
        assert retriever.search('first', 'payment') == []
        assert len(retriever.search('second', 'payment')) == 1
        retriever.delete('second')
        assert retriever.search('second', 'payment') == []


def test_retrieves_one_percent_of_pages_and_preserves_evidence():
    pages = [{'content': f'General notes {i}'} for i in range(100)]
    pages[47] = {'content': 'Cancellation penalty EUR 120', 'page_number': 48,
                 'tables': [[['Penalty', '120']]], 'image': b'original image'}
    with DocumentLoaderRAG(PagesLoader(pages), 'cancellation penalty', max_pages=1) as loader:
        selected = loader.load('document')
        assert len(selected) == 1
        assert selected[0]['page_number'] == 48
        assert selected[0]['tables'] == pages[47]['tables']
        assert selected[0]['image'] == b'original image'
        assert selected[0] is not pages[47]
        assert loader.last_matches[0].page_number == 48
    assert 'page_number' not in pages[0]


def test_extractor_receives_only_retrieved_context():
    class Penalty(Contract):
        amount: int
    pages = [{'content': 'irrelevant opening'}, {'content': 'penalty 120'}, {'content': 'irrelevant ending'}]
    with DocumentLoaderRAG(PagesLoader(pages), 'penalty', max_pages=1) as loader:
        extractor = Extractor(loader)
        extractor.llm = Mock()
        extractor.llm.request.return_value = Penalty(amount=120)
        assert extractor.extract('source.pdf', Penalty).amount == 120
        prompt = str(extractor.llm.request.call_args)
        assert 'penalty 120' in prompt
        assert 'irrelevant opening' not in prompt
        assert 'irrelevant ending' not in prompt
        assert 'page_number: 2' in prompt


def test_no_match_never_silently_falls_back_to_full_document():
    with DocumentLoaderRAG(PagesLoader([{'content': 'invoice'}]), 'absent') as loader:
        with pytest.raises(ValueError, match='No pages matched'):
            loader.load('source')


def test_metadata_tables_are_searchable_and_source_order_is_retained():
    pages = [{'content': '', 'tables': [[['special', '300']]]},
             {'content': 'general'}, {'content': 'special special special'}]
    with DocumentLoaderRAG(PagesLoader(pages), 'special', max_pages=2) as loader:
        assert [p['page_number'] for p in loader.load('source')] == [1, 3]


def test_literal_query_cannot_inject_sql_or_fts_syntax():
    with SQLitePageRetriever() as retriever:
        retriever.index('doc', [{'content': 'invoice reference'}])
        assert retriever.search('doc', 'invoice"; DROP TABLE extractthinker_pages; --')
        assert retriever.search('doc', 'invoice')


@pytest.mark.parametrize('limit', [0, -1, True, 1.5])
def test_invalid_limits_rejected(limit):
    with pytest.raises(ValueError, match='positive integer'):
        DocumentLoaderRAG(PagesLoader([]), 'invoice', max_pages=limit)


def test_shared_retriever_serializes_concurrent_documents():
    with SQLitePageRetriever() as retriever:
        def process(index):
            document = str(index)
            retriever.index(document, [{'content': f'invoice {index}'}])
            return retriever.search(document, 'invoice')[0].page_number
        with ThreadPoolExecutor(max_workers=4) as executor:
            assert list(executor.map(process, range(12))) == [1] * 12


def test_borrowed_retriever_remains_open_after_loader_close():
    with SQLitePageRetriever() as retriever:
        with DocumentLoaderRAG(PagesLoader([{'content': 'invoice'}]), 'invoice', retriever=retriever) as loader:
            loader.load('document')
        retriever.index('other', [{'content': 'still open'}])
        assert retriever.search('other', 'open')


def test_further_page_selection_preserves_original_source_numbers():
    pages = [{'content': 'none'}, {'content': 'invoice'}, {'content': 'invoice'}]
    with DocumentLoaderRAG(PagesLoader(pages), 'invoice', max_pages=2) as loader:
        assert [p['page_number'] for p in loader.load_pages('source', [2, 1])] == [3, 2]

"""Local SQLite FTS5 page retrieval before LLM extraction."""
import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from threading import RLock
from typing import Optional

from extract_thinker.document_loader.document_loader import DocumentLoader


def _page_text(page):
    parts = []
    for key in ('content', 'markdown', 'tables', 'forms'):
        value = page.get(key)
        if value:
            parts.append(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str))
    return '\n'.join(parts)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


@dataclass(frozen=True)
class PageMatch:
    page_number: int
    score: float


class SQLitePageRetriever:
    """Rank indexed page text with SQLite FTS5/BM25; no embedding service required.

    PageMatch numbers are one-based positions in the indexed page list.
    Use a context manager or close() when finished. File databases retain text.
    """
    def __init__(self, database: str = ':memory:'):
        self._lock = RLock()
        self._connection = sqlite3.connect(database, check_same_thread=False)
        try:
            self._connection.execute('CREATE VIRTUAL TABLE IF NOT EXISTS extractthinker_pages USING fts5(document_id UNINDEXED, page_number UNINDEXED, content)')
            self._connection.commit()
        except sqlite3.OperationalError as exc:
            self._connection.close()
            raise RuntimeError('Page retrieval requires SQLite with FTS5 enabled') from exc

    @staticmethod
    def _query(query):
        if not isinstance(query, str):
            raise ValueError('query must be a string containing searchable words')
        terms = list(dict.fromkeys(re.findall(r'\w+', query, flags=re.UNICODE)))
        if not terms:
            raise ValueError('query must contain searchable words')
        # Treat user text as literal terms, not SQL or FTS query syntax.
        return ' OR '.join('"' + term + '"' for term in terms)

    def index(self, document_id, pages):
        if not isinstance(document_id, str) or not document_id:
            raise ValueError('document_id must be a nonempty string')
        if not isinstance(pages, list) or not all(isinstance(page, dict) for page in pages):
            raise ValueError('pages must be a list of page dictionaries')
        rows = [(document_id, number, _page_text(page)) for number, page in enumerate(pages, 1)]
        with self._lock, self._connection:
            self._connection.execute('DELETE FROM extractthinker_pages WHERE document_id = ?', (document_id,))
            self._connection.executemany('INSERT INTO extractthinker_pages(document_id, page_number, content) VALUES (?, ?, ?)', rows)

    def search(self, document_id, query, max_pages=5):
        _positive_integer(max_pages, 'max_pages')
        expression = self._query(query)
        with self._lock:
            rows = self._connection.execute(
                'SELECT page_number, bm25(extractthinker_pages) FROM extractthinker_pages '
                'WHERE extractthinker_pages MATCH ? AND document_id = ? '
                'ORDER BY bm25(extractthinker_pages), CAST(page_number AS INTEGER) LIMIT ?',
                (expression, document_id, max_pages),
            ).fetchall()
        return [PageMatch(int(number), float(score)) for number, score in rows]

    def delete(self, document_id):
        with self._lock, self._connection:
            self._connection.execute('DELETE FROM extractthinker_pages WHERE document_id = ?', (document_id,))

    def close(self):
        with self._lock:
            self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class DocumentLoaderRAG(DocumentLoader):
    """Load a document, retrieve relevant pages, then pass only those to extraction.

    This reduces model input, not the wrapped loader's OCR/parsing work.
    Selection is lexical and may omit relevant pages without matching words.
    """
    def __init__(self, loader: DocumentLoader, query: str, max_pages: int = 5,
                 retriever: Optional[SQLitePageRetriever] = None):
        _positive_integer(max_pages, 'max_pages')
        SQLitePageRetriever._query(query)
        super().__init__()
        self.loader = loader
        self.query = query
        self.max_pages = max_pages
        self._owns_retriever = retriever is None
        self.retriever = retriever if retriever is not None else SQLitePageRetriever()
        self.vision_mode = loader.vision_mode
        self.SUPPORTED_FORMATS = getattr(loader, 'SUPPORTED_FORMATS', [])
        self.last_matches = []

    def can_handle(self, source):
        return self.loader.can_handle(source)

    def can_handle_vision(self, source):
        return self.loader.can_handle_vision(source)

    def can_handle_paginate(self, source):
        return self.loader.can_handle_paginate(source)

    def set_vision_mode(self, enabled=True):
        super().set_vision_mode(enabled)
        self.loader.set_vision_mode(enabled)

    def set_max_image_size(self, size):
        super().set_max_image_size(size)
        self.loader.set_max_image_size(size)

    def load(self, source):
        pages = self.loader.load(source)
        if not isinstance(pages, list) or not all(isinstance(page, dict) for page in pages):
            raise ValueError('Retrieval requires a loader returning page dictionaries')
        # Content identity prevents stale or cross-document page matches.
        text = json.dumps([_page_text(page) for page in pages], ensure_ascii=False)
        document_id = hashlib.sha256(text.encode('utf-8')).hexdigest()
        self.retriever.index(document_id, pages)
        matches = self.retriever.search(document_id, self.query, self.max_pages)
        self.last_matches = matches
        if not matches:
            raise ValueError('No pages matched the retrieval query; broaden the query or use the full document')
        # Model input follows source order, while last_matches retains relevance order.
        return [dict(pages[match.page_number - 1],
                     page_number=pages[match.page_number - 1].get('page_number', match.page_number))
                for match in sorted(matches, key=lambda item: item.page_number)]

    def close(self):
        if self._owns_retriever:
            self.retriever.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

# Retrieve relevant pages before extraction

`DocumentLoaderRAG` wraps an existing loader and uses a local SQLite full-text index to select relevant pages before an LLM request. This is useful when a small part of a long document contains the requested information.

```python
from extract_thinker import (
    Contract, Extractor, DocumentLoaderPyMuPDF, DocumentLoaderRAG,
)

class CancellationTerms(Contract):
    penalty: str

with DocumentLoaderRAG(
    DocumentLoaderPyMuPDF(),
    query="cancellation termination penalty",
    max_pages=3,
) as loader:
    extractor = Extractor(loader)
    extractor.load_llm("your-provider/your-model")
    result = extractor.extract("long-contract.pdf", CancellationTerms)
    print(result)
    print(loader.last_matches)
```

Install the wrapped loader's dependencies (for the example, `pymupdf`). Retrieval itself uses Python's `sqlite3` and requires a SQLite build with [FTS5 support](https://www.sqlite.org/fts5.html). It needs no embedding API or vector database.

## Selection behavior

The index searches page text, Markdown, tables and forms. It ranks matching pages with BM25, keeps up to `max_pages`, and returns them in source order. Images and other page metadata remain attached to the selected pages. `last_matches` retains relevance order; each `PageMatch` contains a one-based position in the original loaded page list and its BM25 score (lower ranks first).

Queries are interpreted as literal words with OR matching; SQL and advanced FTS query syntax are not accepted. Include terms that are likely to appear in the document. Matching is lexical, so synonyms without shared words or content visible only in images can be missed. Use an OCR loader for scans and assess retrieval coverage for your documents. No match raises an error and asks you to broaden the query or use the full document.

This reduces model input, not OCR/parsing work: the underlying loader processes the complete document first. It limits page count, not exact tokens, and it cannot guarantee that selected pages contain every relevant fact. For example, `max_pages=1` selects at most 1% of a 100-page document. Use ordinary extraction when full-document coverage is essential.

`load_pages` on the wrapper further selects positions within the retrieved results. Existing source `page_number` metadata is retained rather than renumbered.

## Persistent SQLite index

By default, each wrapper owns an in-memory index and closes it when the context manager exits. To reuse a file index or index preloaded pages directly:

```python
from extract_thinker import SQLitePageRetriever

with SQLitePageRetriever("document-pages.sqlite") as retriever:
    retriever.index("contract-2026", pages)
    matches = retriever.search("contract-2026", "termination notice", max_pages=5)
    selected = [pages[match.page_number - 1] for match in matches]
    retriever.delete("contract-2026")
```

Indexing an existing document ID atomically replaces its pages. Searches are scoped to that ID. The database stores extracted text locally; it does not store page images or complete metadata. Pass `retriever=retriever` to a wrapper to share it; the wrapper does not close a supplied retriever. Close the retriever yourself or use its context manager.

These APIs are available on main after release 0.1.14.

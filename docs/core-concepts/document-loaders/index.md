# Choose a document loader

A loader turns a file or stream into page dictionaries for extraction. Each page contains `content` and may include images, tables, source page numbers or regions. Available fields depend on the parser and input; inspect a representative document before choosing your extraction strategy.

## Match the loader to the input

| Input or need | Start with | Setup |
| --- | --- | --- |
| Plain text | [DocumentLoaderTxt](txt.md) | Core package |
| Existing page dictionaries | [DocumentLoaderData](data.md) | Core package |
| PDF text layer | [DocumentLoaderPyPdf](pypdf.md) | `pip install pypdf` |
| PDF text, tables and geometry | [DocumentLoaderPyMuPDF](pymupdf.md) | Optional PyMuPDF SDK |
| PDF table layouts | [Camelot or Tabula](pdf-tables.md) | Optional SDK; Tabula also needs Java |
| Local OCR | [Tesseract](tesseract.md), [EasyOCR](easy_ocr.md) or [Docling](docling.md) | Native tools/model assets vary |
| Images for a vision LLM | [DocumentLoaderLLMImage](llm-image.md) | Vision-capable extraction model |
| Managed OCR/document parsing | [Azure](azure-form.md), [AWS](aws-textract.md), [Google](google-document-ai.md), [Adobe](adobe-pdf.md) or [Mistral](mistral-ocr.md) | Provider SDK and credentials |
| Spreadsheets | [Spreadsheet loader](spreadsheet.md) | Optional spreadsheet dependency |
| Office formats | [MarkItDown](markitdown.md) or [Doc2txt](doc2txt.md) | Optional parser dependency |
| Web pages | [Web loader](web-loader.md) | Playwright browser setup |

Optional dependencies are not all installed with the core library. Follow the selected loader's page for supported options and environment requirements. Cloud parsing may upload the source document before the extraction LLM is called.

## Inspect the pages

```python
from io import BytesIO
from extract_thinker import DocumentLoaderPyPdf

loader = DocumentLoaderPyPdf()
pages = loader.load("invoice.pdf")
print(pages[0]["content"])

with open("invoice.pdf", "rb") as source:
    from_stream = loader.load(BytesIO(source.read()))
```

A scanned PDF can have no useful text layer. Use OCR or vision when text extraction alone cannot read the source. Vision rendering does not itself recognize text; it supplies images for the selected model.

## Select source pages

On the updated `main` branch, loaders expose one-based page selection:

```python
pages = loader.load_pages("invoice.pdf", [1, 3])
assert [page["page_number"] for page in pages] == [1, 3]
```

The generic selection API may load the document before filtering. It is not a guarantee of partial parsing. The MCP PDF service selects pages before rendering. See [retrieval](../extractors/retrieval.md) when you want to rank pages by a text query before model calls.

## Compose additional behavior

- [DocumentLoaderRAG](../extractors/retrieval.md) selects pages using SQLite text retrieval.
- [DocumentLoaderMasked](../extractors/masking.md) masks configured entities and patterns in text.
- [DocumentLoaderEvents](events.md) detects page signals or evaluates local rules before invoking callbacks.

Configuration and caching vary by loader. Reusing an instance may reuse cached parsing results; configure its documented cache options when source files change. Consult the individual reference rather than assuming every parser exposes the same options.

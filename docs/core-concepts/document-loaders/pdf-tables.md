# Camelot and Tabula PDF table loaders

These optional adapters extract tables from PDFs and return the same page dictionaries as other ExtractThinker loaders. Each page contains `content` (PDF text), `page_number` (one-based), and `tables` (a list of tables, each a list of rows of strings). Pages with no detected tables remain in the result with `tables=[]`; blank cells and table boundaries are retained.

## Camelot

```bash
python -m pip install camelot-py
```

Install `camelot-py`, not the unrelated package named `camelot`. See the [Camelot API reference](https://camelot-py.readthedocs.io/en/stable/api.html) for parser options and the dependencies of your installed version.

```python
from extract_thinker import DocumentLoaderCamelot, CamelotConfig

loader = DocumentLoaderCamelot(CamelotConfig(flavor="lattice"))
pages = loader.load("invoice.pdf")
for page in pages:
    for table in page["tables"]:
        print(page["page_number"], table)
```

`lattice` is useful for ruled tables; `stream` uses text spacing. `network`, `hybrid`, and `auto` depend on the installed Camelot version. Additional parser options can be passed through `read_pdf_kwargs`, for example `{"split_text": True}`. Source, page range, password and flavor cannot be overridden through that dictionary.

## Tabula

```bash
python -m pip install tabula-py
java -version
```

Install a Java runtime and make `java` available on `PATH`. Installing the Python package alone does not install Java. See the [tabula-py API reference](https://tabula-py.readthedocs.io/en/latest/tabula.html).

```python
from extract_thinker import DocumentLoaderTabula, TabulaConfig

loader = DocumentLoaderTabula(TabulaConfig(
    lattice=True,
    read_pdf_kwargs={"force_subprocess": True},
))
pages = loader.load("invoice.pdf")
```

Use either `lattice=True` for ruled tables or `stream=True` for text layout. `guess` defaults to `True`; `java_options` can configure the runtime. `read_pdf_kwargs` accepts additional parser options, such as `area` or `columns`, but cannot override source, page selection, output format, or explicitly configured settings. The loader requests each page separately because Tabula's table JSON does not consistently carry page identity. This preserves blank pages and multiple tables, with additional invocation overhead on long documents.

## Shared configuration and behavior

Both configurations accept `cache_ttl=300`, `password=None`, and `vision_enabled=False`.

```python
from io import BytesIO

stream = BytesIO(pdf_bytes)
pages = loader.load(stream)
loader.set_vision_mode(True)
loader.set_max_image_size(1024)
selected = loader.load_pages("invoice.pdf", [3, 1])
```

Streams retain their position. Temporary PDFs used by parser backends are deleted after success or failure. Vision mode adds a PNG `image` per page; PDFium reads text and renders images locally. `load_pages` selects after parsing and does not reduce table-extraction work. Parser and Java failures propagate; a failed run is not cached as an empty document.

These adapters do not perform OCR. For scans, use an OCR loader or supply a PDF with a usable text layer. Table quality depends on the chosen parser and document layout. The loaders are available on the repository's main branch after release 0.1.14; install from source until a release containing them is published.

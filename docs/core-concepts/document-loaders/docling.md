# Docling Document Loader

Docling converts documents to Markdown with optional OCR and table structure detection. The SDK is optional; importing ExtractThinker does not load Docling.

## Installation and dependency repair

Install Docling into the same environment as ExtractThinker:

```bash
python -m pip install docling
python -m pip check
```

An error such as `No module named 'docling_core.types.doc.page'` comes from Docling's dependency imports. It usually indicates an inconsistent `docling` / `docling-core` installation. The loader now preserves the original exception and reports installed package versions.

Resolve the SDK and its dependencies together:

```bash
python -m pip install --upgrade --upgrade-strategy eager docling
python -m pip check
```

If the import still fails, create a fresh virtual environment and install ExtractThinker and Docling together. Avoid upgrading or pinning `docling-core` independently. Current Docling releases require Python 3.10 or newer; ExtractThinker's Python 3.9 support does not imply that every optional SDK supports it. See [Docling installation](https://docling-project.github.io/docling/getting_started/installation/).

## Basic usage

```python
from extract_thinker import DocumentLoaderDocling, DoclingConfig

loader = DocumentLoaderDocling(DoclingConfig(
    ocr_enabled=False,
    table_structure_enabled=True,
    do_cell_matching=True,
))
pages = loader.load("invoice.pdf")
for page in pages:
    print(page["page_number"], page["markdown"])
```

Each result has `content` and `markdown` containing the same Markdown, a one-based `page_number`, and `image` (bytes when rendered, otherwise `None`). Tables appear in the Markdown; this adapter does not return a separate `tables` list. Paginated documents retain their pages, including PDFs loaded from URLs. Formats without page information return one result for the entire document.

PDF, DOCX, PPTX, XLSX, HTML, Markdown, AsciiDoc, supported XML formats, plain text, and common raster images are accepted by this adapter. Actual parsing depends on the installed Docling version and its backend dependencies. A `BytesIO` source is treated as a PDF; use a path with the correct extension for other formats. The caller's stream position is preserved.

## Configuration

| Option | Default | Effect |
| --- | --- | --- |
| `cache_ttl` | `300` | Cache lifetime in seconds |
| `ocr_enabled` | `False` | Enable OCR in the PDF pipeline |
| `table_structure_enabled` | `True` | Enable table structure detection |
| `force_full_page_ocr` | `False` | Apply OCR across the entire page |
| `do_cell_matching` | `True` | Match table cells to source content |
| `format_options` | `None` | Supply Docling format option objects; overrides the simple pipeline settings |
| `content` | `None` | Optional initial content |

When using `format_options`, pass SDK objects rather than arbitrary dictionaries:

```python
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.document_converter import PdfFormatOption
from extract_thinker import DocumentLoaderDocling, DoclingConfig

pipeline = PdfPipelineOptions(do_ocr=True)
pipeline.ocr_options = TesseractCliOcrOptions(
    tesseract_cmd="tesseract",
    force_full_page_ocr=True,
)
loader = DocumentLoaderDocling(DoclingConfig(format_options={
    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline),
}))
```

Tesseract must be installed separately for this example. PDF pipelines may download model artifacts on first use. Custom OCR configuration belongs in the SDK's `ocr_options`; `DoclingConfig` does not accept `tesseract_cmd` directly.

## Images and selected pages

```python
loader.set_vision_mode(True)
pages = loader.load_pages("invoice.pdf", [1, 3])
```

Page images are rendered once per uncached load and matched to their source page. URL documents without page information expose captured images in `images`. Rendering requires the relevant PDF/image/browser support. Page selection occurs after conversion and does not reduce upstream parsing work.

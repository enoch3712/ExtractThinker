# Adobe PDF Extract loader

`DocumentLoaderAdobePDF` uses Adobe's PDF Services SDK to extract text and tables from native or scanned PDFs. Loading a document uploads it to Adobe's service and uses the account's API allowance. Credentials and service access are required. See [Adobe's Python quickstart](https://developer.adobe.com/document-services/docs/overview/pdf-extract-api/quickstarts/extract-pdf/python/) for account setup.

## Installation and usage

```bash
python -m pip install pdfservices-sdk
export PDF_SERVICES_CLIENT_ID="your-client-id"
export PDF_SERVICES_CLIENT_SECRET="your-client-secret"
```

Use environment variables or your application's secret manager for credentials. Current SDK versions require Python 3.10 or newer.

```python
from extract_thinker import DocumentLoaderAdobePDF, AdobePDFConfig

loader = DocumentLoaderAdobePDF(AdobePDFConfig(
    extract_tables=True,
    include_bbox=True,
))
pages = loader.load("invoice.pdf")
for page in pages:
    print(page["page_number"], page["content"])
    print(page["tables"])
```

You can also pass `client_id` and `client_secret` in `AdobePDFConfig`, or supply an already configured `PDFServices` instance with `DocumentLoaderAdobePDF(client=client)`. Configuration reprs omit credentials; avoid logging your own credential objects.

## Output and configuration

- `content`: extracted text in the order of Adobe's elements for the page.
- `page_number`: one-based source page number. Blank pages remain in the output.
- `tables`: separate tables containing rows of strings, read from CSV files in Adobe's result. Headers and blank cells are retained. A table is associated with the page of the element that references its CSV.
- `regions`: included when `include_bbox=True` and Adobe supplies element bounds and page dimensions. Coordinates are normalized to the page with a top-left origin.
- `image`: PNG rendering of the original page when `vision_enabled=True` or `set_vision_mode(True)` is used.

`extract_tables` defaults to `True`, `include_bbox` and `vision_enabled` to `False`, and `cache_ttl` to 300 seconds. The SDK receives requests for text and optional tables, with CSV as the table format. Image rendering happens locally with PDFium.

Both local PDF paths and `BytesIO` streams are accepted. Streams retain their position; temporary PDFs are cleaned up after use. The input must be an unencrypted PDF. `load_pages(source, [1, 3])` selects after the complete service request and does not reduce service usage. Provider errors, invalid page references, or malformed result archives raise instead of returning a partial success.

The adapter is available on main after release 0.1.14. Its tests cover SDK job construction with a fake service and structured-result mapping; automated tests do not exercise Adobe's live extraction service.

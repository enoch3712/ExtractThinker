"""Adobe PDF Extract adapter using the optional PDF Services SDK."""
import csv
import json
import os
from dataclasses import dataclass, field
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Optional
from zipfile import ZipFile

from extract_thinker.document_loader.pdf_table_loader import PDFTableLoader
from extract_thinker.models.document_evidence import BoundingBox, DocumentRegion


@dataclass
class AdobePDFConfig:
    client_id: Optional[str] = field(default=None, repr=False)
    client_secret: Optional[str] = field(default=None, repr=False)
    cache_ttl: int = 300
    vision_enabled: bool = False
    extract_tables: bool = True
    include_bbox: bool = False
    # Adobe's service accepts unencrypted PDFs; PDFium uses no password here.
    password: Optional[str] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.cache_ttl <= 0:
            raise ValueError('cache_ttl must be positive')


class DocumentLoaderAdobePDF(PDFTableLoader):
    """Upload a PDF to Adobe Extract and map its structured result to pages.

    Supply a PDFServices client or configure PDF_SERVICES_CLIENT_ID and
    PDF_SERVICES_CLIENT_SECRET. Loading sends the document to Adobe's service.
    """
    def __init__(self, config: Optional[AdobePDFConfig] = None, client: Any = None):
        config = config or AdobePDFConfig()
        if client is None:
            try:
                from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
                from adobe.pdfservices.operation.pdf_services import PDFServices
            except ImportError as exc:
                raise ImportError('Adobe PDF loader requires `pip install pdfservices-sdk`.') from exc
            client_id = config.client_id or os.getenv('PDF_SERVICES_CLIENT_ID')
            client_secret = config.client_secret or os.getenv('PDF_SERVICES_CLIENT_SECRET')
            if not client_id or not client_secret:
                raise ValueError('Set both PDF_SERVICES_CLIENT_ID and PDF_SERVICES_CLIENT_SECRET, or pass credentials in AdobePDFConfig')
            client = PDFServices(credentials=ServicePrincipalCredentials(
                client_id=client_id, client_secret=client_secret))
        self.client = client
        super().__init__(config)

    def _extract_archive(self, pdf_bytes):
        try:
            from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
            from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
            from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
            from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import ExtractPDFParams
            from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.table_structure_type import TableStructureType
            from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult
        except ImportError as exc:
            raise ImportError('Adobe PDF loader requires `pip install pdfservices-sdk`.') from exc
        elements = [ExtractElementType.TEXT]
        if self.config.extract_tables:
            elements.append(ExtractElementType.TABLES)
        asset = self.client.upload(input_stream=pdf_bytes, mime_type=PDFServicesMediaType.PDF)
        params = ExtractPDFParams(elements_to_extract=elements, table_structure_type=TableStructureType.CSV)
        location = self.client.submit(ExtractPDFJob(input_asset=asset, extract_pdf_params=params))
        response = self.client.get_job_result(location, ExtractPDFResult)
        output_asset = response.get_result().get_resource()
        return self.client.get_content(output_asset).get_input_stream()

    def _read_tables(self, path, pages):
        archive = self._extract_archive(Path(path).read_bytes())
        self._apply_archive(archive, pages)

    def _apply_archive(self, archive, pages):
        with ZipFile(BytesIO(archive)) as bundle:
            document = json.loads(bundle.read('structuredData.json'))
            dimensions = {page['page_number']: page for page in document.get('pages', [])}
            texts = [[] for _ in pages]
            seen_tables = set()
            for element in document.get('elements', []):
                number = element.get('Page')
                if number is None and not element.get('Text') and not element.get('filePaths'):
                    continue
                if isinstance(number, bool) or not isinstance(number, int) or not 0 <= number < len(pages):
                    raise ValueError(f'Adobe returned an invalid source page: {number}')
                page = pages[number]
                text = element.get('Text')
                if text:
                    texts[number].append(text)
                    if self.config.include_bbox and element.get('Bounds') and number in dimensions:
                        bounds = element['Bounds']
                        size = dimensions[number]
                        width, height = size['width'], size['height']
                        if width <= 0 or height <= 0:
                            raise ValueError('Adobe returned invalid page dimensions')
                        box = BoundingBox(page=number + 1, x0=bounds[0] / width,
                                          y0=1 - bounds[3] / height, x1=bounds[2] / width,
                                          y1=1 - bounds[1] / height)
                        page.setdefault('regions', []).append(DocumentRegion(text=text, bounding_box=box).model_dump())
                if self.config.extract_tables:
                    for name in element.get('filePaths', []):
                        if name.lower().endswith('.csv') and name not in seen_tables:
                            rows = list(csv.reader(StringIO(bundle.read(name).decode('utf-8-sig'))))
                            page['tables'].append(rows)
                            seen_tables.add(name)
            for page, fragments in zip(pages, texts):
                page['content'] = '\n'.join(fragments)

"""Shared PDF metadata and temporary-file handling for optional table adapters."""
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock

import pypdfium2 as pdfium
from cachetools import cachedmethod
from cachetools.keys import hashkey
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader

_TABLE_PDF_LOCK = RLock()


class PDFTableLoader(CachedDocumentLoader):
    SUPPORTED_FORMATS = ['pdf']

    def __init__(self, config):
        self.config = config
        super().__init__(cache_ttl=config.cache_ttl)
        self.vision_mode = config.vision_enabled

    @contextmanager
    def _pdf_path(self, source):
        if isinstance(source, str):
            yield source
        else:
            with TemporaryDirectory(prefix='extractthinker-tables-') as directory:
                path = Path(directory) / 'document.pdf'
                path.write_bytes(source.getvalue())
                yield str(path)

    def _pages(self, source):
        # PDFium's native API is not thread-safe.
        with _TABLE_PDF_LOCK:
            document = pdfium.PdfDocument(
                source if isinstance(source, str) else source.getvalue(),
                password=self.config.password,
            )
            try:
                pages = []
                for index in range(len(document)):
                    page = document[index]
                    try:
                        text = page.get_textpage()
                        try:
                            content = text.get_text_range()
                        finally:
                            text.close()
                        data = dict(content=content, page_number=index + 1, tables=[])
                        if self.vision_mode:
                            bitmap = page.render(scale=150 / 72)
                            try:
                                image = bitmap.to_pil()
                                try:
                                    resized = self._resize_if_needed(image)
                                    try:
                                        output = BytesIO()
                                        resized.save(output, format='PNG')
                                        data['image'] = output.getvalue()
                                    finally:
                                        if resized is not image:
                                            resized.close()
                                finally:
                                    image.close()
                            finally:
                                bitmap.close()
                        pages.append(data)
                    finally:
                        page.close()
                return pages
            finally:
                document.close()

    @cachedmethod(cache=lambda self: self.cache, key=lambda self, source: hashkey(
        source if isinstance(source, str) else source.getvalue(), self.vision_mode,
        repr(self.config), self.config.password, self.max_image_size,
    ))
    def load(self, source):
        if not self.can_handle(BytesIO(source.getvalue()) if isinstance(source, BytesIO) else source):
            raise ValueError('Table loaders require a local PDF path or BytesIO stream')
        pages = self._pages(source)
        with self._pdf_path(source) as path:
            self._read_tables(path, pages)
        return pages

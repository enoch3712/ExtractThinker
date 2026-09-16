"""Optional PyMuPDF adapter with text, tables, page images and source regions."""
from dataclasses import dataclass
from io import BytesIO
from operator import attrgetter
from threading import RLock
from typing import Any, Dict, List, Optional, Union

from cachetools import cachedmethod
from cachetools.keys import hashkey

from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader
from extract_thinker.models.document_evidence import BoundingBox, DocumentRegion

_PYMUPDF_LOCK = RLock()


@dataclass
class PyMuPDFConfig:
    cache_ttl: int = 300
    password: Optional[str] = None
    extract_tables: bool = False
    include_bbox: bool = False
    vision_enabled: bool = False
    dpi: int = 150

    def __post_init__(self):
        if self.cache_ttl <= 0:
            raise ValueError("cache_ttl must be positive")
        if isinstance(self.dpi, bool) or not isinstance(self.dpi, int) or self.dpi <= 0:
            raise ValueError("dpi must be a positive integer")


class DocumentLoaderPyMuPDF(CachedDocumentLoader):
    """Read PDF text without OCR; install `pymupdf` separately to use this loader."""

    SUPPORTED_FORMATS = ["pdf"]

    def __init__(self, config: Optional[PyMuPDFConfig] = None):
        try:
            import pymupdf
        except ImportError as exc:
            raise ImportError("PyMuPDF loader requires `pip install pymupdf`.") from exc
        self._pymupdf = pymupdf
        self.config = config or PyMuPDFConfig()
        super().__init__(cache_ttl=self.config.cache_ttl)
        self.vision_mode = self.config.vision_enabled

    @cachedmethod(cache=attrgetter("cache"), key=lambda self, source: hashkey(
        source if isinstance(source, str) else source.getvalue(), self.vision_mode,
        self.config.include_bbox, self.config.extract_tables, self.config.dpi,
        self.max_image_size,
    ))
    def load(self, source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        # PyMuPDF does not support concurrent native calls from Python threads.
        with _PYMUPDF_LOCK:
            return self._load(source)

    def _load(self, source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        if not self.can_handle(source):
            raise ValueError("PyMuPDF loader requires a PDF path or BytesIO stream")
        document = (self._pymupdf.open(source) if isinstance(source, str)
                    else self._pymupdf.open(stream=source.getvalue(), filetype="pdf"))
        with document:
            if document.needs_pass and not document.authenticate(self.config.password or ""):
                raise ValueError("A valid password is required for this PDF")
            pages = []
            for index, page in enumerate(document):
                data = {"content": page.get_text("text", sort=True), "page_number": index + 1}
                if self.config.extract_tables:
                    data["tables"] = [table.extract() for table in page.find_tables().tables]
                if self.config.include_bbox:
                    regions = []
                    for block in page.get_text("blocks", sort=True):
                        if block[6] != 0:  # Ignore image blocks.
                            continue
                        # PyMuPDF text coordinates are unrotated; align to rendered page.
                        rect = self._pymupdf.Rect(block[:4]) * page.rotation_matrix
                        bounds = page.rect
                        clamp = lambda value: max(0.0, min(1.0, value))
                        box = BoundingBox(
                            page=index + 1,
                            x0=clamp(rect.x0 / bounds.width),
                            y0=clamp(rect.y0 / bounds.height),
                            x1=clamp(rect.x1 / bounds.width),
                            y1=clamp(rect.y1 / bounds.height),
                        )
                        regions.append(DocumentRegion(text=block[4], bounding_box=box).model_dump())
                    data["regions"] = regions
                if self.vision_mode:
                    pixmap = page.get_pixmap(dpi=self.config.dpi, alpha=False)
                    image_bytes = pixmap.tobytes("png")
                    if self.max_image_size is not None:
                        from PIL import Image
                        with Image.open(BytesIO(image_bytes)) as image:
                            resized = self._resize_if_needed(image)
                            try:
                                output = BytesIO()
                                resized.save(output, format="PNG")
                                image_bytes = output.getvalue()
                            finally:
                                if resized is not image:
                                    resized.close()
                    data["image"] = image_bytes
                pages.append(data)
            return pages

import io
from io import BytesIO
from typing import Any, Dict, List, Union
from PIL import Image
import numpy as np
from dataclasses import dataclass, field
from cachetools import cachedmethod, TTLCache
from cachetools.keys import hashkey
import easyocr
import pdf2image

from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader
from extract_thinker.utils import is_pdf_stream


@dataclass
class EasyOCRConfig:
    """Configuration for EasyOCR loader.

    Args:
        lang_list: the languages to use for OCR
        gpu: whether to use GPU for OCR
        download_enabled: whether to download the models if they are not found
        cache_ttl: time-to-live for OCR result caching, in seconds
    """
    lang_list: List[str] = field(default_factory=lambda: ['en'])
    gpu: bool = True
    download_enabled: bool = True
    cache_ttl: int = 300  # 🆕 TTL for cache in seconds (default: 10 minutes)

    def __post_init__(self):
        if not self.lang_list:
            self.lang_list = ['en']
        self.reader = easyocr.Reader(
            lang_list=self.lang_list,
            gpu=self.gpu,
            download_enabled=self.download_enabled
        )


class DocumentLoaderEasyOCR(CachedDocumentLoader):
    SUPPORTED_FORMATS = ["pdf", "png", "jpg", "jpeg"]

    def __init__(self, config: EasyOCRConfig):
        super().__init__()
        self.config = config
        self.vision_mode = False  # easyocr is mostly normal text extraction
        self.cache = TTLCache(maxsize=128, ttl=self.config.cache_ttl)  # 🆕 TTL-based cache

    def set_vision_mode(self, enabled: bool) -> None:
        """Enable or disable vision mode."""
        self.vision_mode = enabled

    def can_handle_vision(self, source: Union[str, BytesIO]) -> bool:
        """Check if vision mode is supported for this source."""
        return self.can_handle(source)

    def can_handle(self, source: Union[str, BytesIO]) -> bool:
        if isinstance(source, BytesIO):
            return True
        if isinstance(source, str):
            if '.' not in source:
                return False
            ext = source.split('.')[-1].lower()
            return ext in self.SUPPORTED_FORMATS
        return False

    def load(self, source: Union[str, BytesIO]) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        file_bytes = source.getvalue() if isinstance(source, BytesIO) else open(source, 'rb').read()
        page_results = self.load_document(file_bytes)

        if self.vision_mode and self.can_handle_vision(source):
            images = self._convert_to_images(file_bytes)
            image_bytes_dict = {}
            for i, img in enumerate(images):
                buf = BytesIO()
                img.save(buf, format="PNG")
                image_bytes_dict[i] = buf.getvalue()
            return {
                "pages": page_results,
                "images": image_bytes_dict
            }
        return page_results

    @cachedmethod(lambda self: self.cache, key=lambda self, file_bytes, **_: hashkey(file_bytes))
    def load_document(self, file_bytes: bytes, **kwargs) -> List[List[Dict[str, Any]]]:
        """
        Process document bytes and return OCR results as list of pages.
        Each page is a list with bbox, text, and probability.
        """
        images = self._convert_to_images(file_bytes)
        results = []
        for img in images:
            ocr_result = self.config.reader.readtext(np.array(img))
            page_data = []
            for bbox, text, prob in ocr_result:
                page_data.append({
                    "bbox": bbox,
                    "text": text,
                    "probability": prob
                })
            results.append(page_data)
        return results

    def _convert_to_images(self, file_bytes: bytes) -> List[Image.Image]:
        try:
            if is_pdf_stream(file_bytes):
                return pdf2image.convert_from_bytes(file_bytes)
            else:
                img = Image.open(BytesIO(file_bytes)).convert("RGB")
                return [img]
        except Exception as e:
            if file_bytes.startswith(b'%PDF'):
                return pdf2image.convert_from_bytes(file_bytes)
            raise RuntimeError(f"Failed to process file: {e}")

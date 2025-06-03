from typing import Any, Dict, List
from PIL import Image
import numpy as np
from dataclasses import dataclass, field
from cachetools import cachedmethod, TTLCache
from cachetools.keys import hashkey
from operator import attrgetter
import easyocr

from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader


@dataclass
class EasyOCRConfig:
    """Configuration for EasyOCR loader.

    Args:
        lang_list: List of languages to use for OCR
        gpu: Whether to use GPU acceleration
        download_enabled: Whether to download models automatically
        cache_ttl: Time-to-live for cache in seconds
    """
    lang_list: List[str] = field(default_factory=lambda: ['en'])
    gpu: bool = True
    download_enabled: bool = True
    cache_ttl: int = 300

    def __post_init__(self):
        if not self.lang_list:
            self.lang_list = ['en']
        self.reader = easyocr.Reader(
            lang_list=self.lang_list,
            gpu=self.gpu,
            download_enabled=self.download_enabled,
        )


class DocumentLoaderEasyOCR(CachedDocumentLoader):
    SUPPORTED_FORMATS = ["png", "jpg", "jpeg", "tiff", "tif", "webp"]

    def __init__(self, config: EasyOCRConfig):
        super().__init__()
        self.config = config
        self.cache = TTLCache(maxsize=128, ttl=self.config.cache_ttl)

    def can_handle(self, source: str) -> bool:
        if not isinstance(source, str) or '.' not in source:
            return False
        ext = source.split('.')[-1].lower()
        return ext in self.SUPPORTED_FORMATS

    @cachedmethod(cache=attrgetter('cache'), key=lambda _, path: hashkey(path))
    def load(self, image_path: str) -> List[List[Dict[str, Any]]]:
        """Load and process an image using EasyOCR.

        Args:
            image_path: Path to the image file

        Returns:
            List of pages, where each page contains a list of OCR results.
            Each OCR result is a dictionary with 'text', 'probability', and 'bbox' keys.
        """
        with Image.open(image_path).convert("RGB") as img:
            ocr_result = self.config.reader.readtext(np.array(img))
        page_data = []
        for bbox, text, prob in ocr_result:
            page_data.append({
                "bbox": bbox,
                "text": text,
                "probability": prob
            })
        return [page_data]

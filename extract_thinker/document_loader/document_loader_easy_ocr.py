from typing import Any, Dict, List, Union
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
        lang_list: list of languages to use 
        gpu: whether to use gpu
        download_enabled: whether to download models
        cache_ttl: time to live for cache
        
    
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
    SUPPORTED_FORMATS = ["png", "jpg", "jpeg", "tiff", "webp"]

    def __init__(self, config: EasyOCRConfig):
        super().__init__()
        self.config = config
        self.cache = TTLCache(maxsize=128, ttl=self.config.cache_ttl)

    def can_handle(self, source: str) -> bool:
        if not isinstance(source, str) or '.' not in source:
            return False
        ext = source.split('.')[-1].lower()
        return ext in self.SUPPORTED_FORMATS

    def load(self, source: str) -> List[List[Dict[str, Any]]]:
        return self.load_document(source)

    @cachedmethod(cache=attrgetter('cache'), key=lambda _, path: hashkey(path))
    def load_document(self, image_path: str) -> List[List[Dict[str, Any]]]:
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

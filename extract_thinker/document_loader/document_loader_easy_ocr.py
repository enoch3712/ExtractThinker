from typing import Any, Dict, List, Union
from io import BytesIO
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
        lang_list: List of languages to use for OCR. Defaults to ['en'].
        gpu: Whether to use GPU acceleration. Defaults to True.
        download_enabled: Whether to download models automatically. Defaults to True.
        cache_ttl: Time-to-live for cache in seconds. Defaults to 300.
    """
    lang_list: List[str] = field(default_factory=lambda: ['en'])
    gpu: bool = True
    download_enabled: bool = True
    cache_ttl: int = 300

    def __post_init__(self):
        """Initialize EasyOCR reader with configuration settings and validation."""
        if not self.lang_list:
            raise ValueError("lang_list must contain at least one language code.")
        if self.cache_ttl <= 0:
            raise ValueError("cache_ttl must be positive.")

        self.reader = easyocr.Reader(
            lang_list=self.lang_list,
            gpu=self.gpu,
            download_enabled=self.download_enabled,
        )


class DocumentLoaderEasyOCR(CachedDocumentLoader):
    SUPPORTED_FORMATS = ["png", "jpg", "jpeg", "tiff", "tif", "webp"]

    def __init__(self, config: EasyOCRConfig):
        """Initialize the EasyOCR document loader.

        Args:
            config: Configuration object for EasyOCR settings
        """
        super().__init__()
        self.config = config
        self.cache = TTLCache(maxsize=128, ttl=self.config.cache_ttl)
        self.vision_mode = False

    def can_handle(self, source: Union[str, BytesIO]) -> bool:
        """Check if the loader can handle the given source.

        Args:
            source: Path to a file or BytesIO stream

        Returns:
            bool: True if source is supported, False otherwise
        """
        # Check if source is a BytesIO stream
        if isinstance(source, BytesIO):
            return True
        # Check if source is a file path and has a valid extension
        if isinstance(source, str) and '.' in source:
             # Extract the file extension (after the last '.') and convert to lowercase
            ext = source.split('.')[-1].lower()
            return ext in self.SUPPORTED_FORMATS
        return False

    @cachedmethod(cache=attrgetter('cache'), key=lambda self, source: hashkey(source) if isinstance(source, str) else None)
    def load(self, source: Union[str, BytesIO]) -> List[List[Dict[str, Any]]]:
        """Load and process an image (file path or BytesIO) using EasyOCR.

        Args:
            source: Image file path or in-memory image stream (BytesIO)

        Returns:
            List of pages, where each page contains a list of OCR results.
            Each OCR result is a dictionary with:
                - text: The extracted text
                - probability: Confidence score
                - bbox: Bounding box coordinates
        """
         # Convert image from file path into numpy array
        if isinstance(source, str):
            with Image.open(source).convert("RGB") as img:
                image_array = np.array(img)
         # Convert image from bytes stream into numpy array
        elif isinstance(source, BytesIO):
            source.seek(0)
            with Image.open(source).convert("RGB") as img:
                image_array = np.array(img)
        else:
            raise ValueError("Unsupported source type. Expected str or BytesIO.")

        ocr_result = self.config.reader.readtext(image_array)
        # Loop through OCR results and structure them into a dictionary format
        page_data = []
        for bbox, text, prob in ocr_result:
            page_data.append({
                "bbox": bbox,
                "text": text,
                "probability": prob
            })
        return [page_data]

    def can_handle_vision(self, source: Union[str, BytesIO]) -> bool:
        """EasyOCR currently doesn't support vision mode in this loader."""
        return False

    def set_vision_mode(self, enabled: bool = True):
        """Disable vision mode, not supported here."""
        if enabled:
            raise ValueError("Vision mode is not supported in EasyOCR loader.")
      
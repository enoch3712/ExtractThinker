from typing import Any, Dict, List, Union
import os
from io import BytesIO
from PIL import Image
import numpy as np
import threading
from queue import Queue
from dataclasses import dataclass, field
from cachetools import cachedmethod, TTLCache
from cachetools.keys import hashkey
from operator import attrgetter

from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader
from extract_thinker.utils import is_pdf_stream


@dataclass
class EasyOCRConfig:
    """Configuration for EasyOCR loader.

    Args:
        lang_list: List of languages to use for OCR. Defaults to ['en'].
        gpu: Whether to use GPU acceleration. Defaults to True.
        download_enabled: Whether to download models automatically. Defaults to True.
        cache_ttl: Time-to-live for cache in seconds. Defaults to 300.
        include_bbox: Whether to include bounding box data in the output. Defaults to False.
    """
    lang_list: List[str] = field(default_factory=lambda: ['en'])
    gpu: bool = True
    download_enabled: bool = True
    cache_ttl: int = 300
    include_bbox: bool = False

    def __post_init__(self):
        """Initialize EasyOCR reader with configuration settings and validation."""
        if not self.lang_list:
            raise ValueError("lang_list must contain at least one language code.")
        if self.cache_ttl <= 0:
            raise ValueError("cache_ttl must be positive.")

        try:
            import easyocr
        except ImportError:
            raise ImportError(
                "The 'easyocr' package is required for this loader but is not installed. "
                "Please install it with: pip install easyocr"
            )

        self.reader = easyocr.Reader(
            lang_list=self.lang_list,
            gpu=self.gpu,
            download_enabled=self.download_enabled,
        )


class DocumentLoaderEasyOCR(CachedDocumentLoader):
    SUPPORTED_FORMATS = ["png", "jpg", "jpeg", "tiff", "tif", "webp", "pdf"]

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

    @cachedmethod(cache=attrgetter('cache'), key=lambda self, source: hashkey(source if isinstance(source, str) else source.getvalue()))
    def load(self, source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        """Load and process an image or PDF (file path or BytesIO) using EasyOCR.

        Args:
            source: Image or PDF file path or in-memory stream (BytesIO)

        Returns:
            List of pages, where each page is a dictionary.
            Each dictionary contains:
                - 'content': The extracted text as a single string.
                - 'detail': (if include_bbox is True) A list of OCR results with bbox, text, and probability.
        """
        if not self.can_handle(source):
            raise ValueError(f"Cannot handle source: {source}")

        try:
            if is_pdf_stream(source):
                return self._process_pdf(source)
            else:
                return self._process_single_image(source)
        except Exception as e:
            raise ValueError(f"Error processing document: {str(e)}")

    def _process_ocr_result(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Runs OCR on a single image array and formats the result."""
        ocr_result = self.config.reader.readtext(image_array)

        # Combine text from all bounding boxes
        full_text = " ".join([text for _, text, _ in ocr_result])

        page_data = {
            "content": full_text
        }

        if self.config.include_bbox:
            page_data["detail"] = [
                {"bbox": bbox, "text": text, "probability": prob}
                for bbox, text, prob in ocr_result
            ]

        return page_data

    def _process_single_image(self, source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        """Process a single image file."""
        if isinstance(source, str):
            with Image.open(source).convert("RGB") as img:
                image_array = np.array(img)
        elif isinstance(source, BytesIO):
            source.seek(0)
            with Image.open(source).convert("RGB") as img:
                image_array = np.array(img)
        else:
            raise ValueError("Unsupported source type. Expected str or BytesIO.")

        page_result = self._process_ocr_result(image_array)
        return [page_result]

    def _process_pdf(self, pdf_source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        """Process a PDF document by converting to images and performing OCR."""
        try:
            if isinstance(pdf_source, str):
                with open(pdf_source, 'rb') as f:
                    images_dict = self.convert_to_images(f)
            else:
                pdf_source.seek(0)
                images_dict = self.convert_to_images(pdf_source)

            if not images_dict:
                raise ValueError("No images were extracted from PDF")

            # Process each page in parallel
            pages = self._process_images_parallel(images_dict)
            return pages

        except Exception as e:
            raise ValueError(f"Error processing PDF: {str(e)}")

    def _process_images_parallel(self, images_dict: Dict[int, bytes]) -> List[Dict[str, Any]]:
        """Process multiple images in parallel using threads."""
        input_queue = Queue()
        output_queue = Queue()

        # Add all images to the input queue
        for page_num, image_bytes in images_dict.items():
            input_queue.put((page_num, image_bytes))

        # Create worker threads
        threads = []
        num_threads = min(4, len(images_dict))
        for _ in range(num_threads):
            t = threading.Thread(target=self._worker, args=(input_queue, output_queue))
            t.start()
            threads.append(t)

        # Wait for all images to be processed
        input_queue.join()

        # Stop workers
        for _ in range(num_threads):
            input_queue.put(None)
        for t in threads:
            t.join()

        # Collect results and sort by page number
        results = []
        while not output_queue.empty():
            results.append(output_queue.get())
        
        results.sort(key=lambda x: x[0])  # Sort by page number
        
        # Return just the page data
        return [page_data for _, page_data in results]

    def _worker(self, input_queue: Queue, output_queue: Queue) -> None:
        """Worker thread for parallel image processing."""
        while True:
            item = input_queue.get()
            if item is None:
                break

            page_num, image_bytes = item
            try:
                with Image.open(BytesIO(image_bytes)).convert("RGB") as img:
                    image_array = np.array(img)
                
                page_result = self._process_ocr_result(image_array)
                output_queue.put((page_num, page_result))
            except Exception as e:
                error_message = {"content": f"Error processing page {page_num}: {e}"}
                if self.config.include_bbox:
                    error_message["detail"] = []
                output_queue.put((page_num, error_message))
            finally:
                input_queue.task_done()

    def can_handle_vision(self, source: Union[str, BytesIO]) -> bool:
        """EasyOCR currently doesn't support vision mode in this loader."""
        return False

    def set_vision_mode(self, enabled: bool = True):
        """Disable vision mode, not supported here."""
        if enabled:
            raise ValueError("Vision mode is not supported in EasyOCR loader.")
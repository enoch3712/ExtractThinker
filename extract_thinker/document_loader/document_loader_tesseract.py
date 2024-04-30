from io import BytesIO
from operator import attrgetter
import os
from typing import Union
from PIL import Image
import pytesseract

from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader
from ..utils import get_image_type

from cachetools import cachedmethod
from cachetools.keys import hashkey

SUPPORTED_IMAGE_FORMATS = ["jpeg", "png", "bmp", "tiff"]


class DocumentLoaderTesseract(CachedDocumentLoader):
    def __init__(self, tesseract_cmd, isContainer=False, content=None, cache_ttl=300):
        super().__init__(content, cache_ttl)
        self.tesseract_cmd = tesseract_cmd
        if isContainer:
            # docker path to tesseract
            self.tesseract_cmd = os.environ.get("TESSERACT_PATH", "tesseract")
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        if not os.path.isfile(self.tesseract_cmd):
            raise Exception(f"Tesseract not found at {self.tesseract_cmd}")

    @cachedmethod(cache=attrgetter('cache'), key=lambda self, file_path: hashkey(file_path))
    def load_content_from_file(self, file_path: str) -> Union[str, object]:
        try:
            file_type = get_image_type(file_path)
            if file_type in SUPPORTED_IMAGE_FORMATS:
                image = Image.open(file_path)
                raw_text = str(pytesseract.image_to_string(image))
                self.content = raw_text
                return self.content
            else:
                raise Exception(f"Unsupported file type: {file_path}")
        except Exception as e:
            raise Exception(f"Error processing file: {e}") from e

    @cachedmethod(cache=attrgetter('cache'), key=lambda self, stream: hashkey(id(stream)))
    def load_content_from_stream(self, stream: Union[BytesIO, str]) -> Union[str, object]:
        try:
            file_type = get_image_type(stream)
            if file_type in SUPPORTED_IMAGE_FORMATS:
                image = Image.open(stream)
                raw_text = str(pytesseract.image_to_string(image))
                self.content = raw_text
                return self.content
            else:
                raise Exception(f"Unsupported stream type: {stream}")
        except Exception as e:
            raise Exception(f"Error processing stream: {e}") from e

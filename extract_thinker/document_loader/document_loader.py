from abc import ABC, abstractmethod
import io
from PIL import Image
from io import BytesIO
import pypdfium2 as pdfium
import concurrent.futures
from typing import Any, Dict, List, Union
from cachetools import TTLCache
import os
from extract_thinker.utils import get_file_extension

class DocumentLoader(ABC):
    def __init__(self, content: Any = None, cache_ttl: int = 300):
        self.content = content
        self.file_path = None
        self.cache = TTLCache(maxsize=100, ttl=cache_ttl)

    def can_handle(self, source: Union[str, BytesIO]) -> bool:
        """
        Checks if the loader can handle the given source.
        
        Args:
            source: Either a file path (str) or a BytesIO stream
            
        Returns:
            bool: True if the loader can handle the source, False otherwise
        """
        try:
            if isinstance(source, str):
                return self._can_handle_file_path(source)
            elif isinstance(source, BytesIO):
                return self._can_handle_stream(source)
            return False
        except Exception:
            return False

    def _can_handle_file_path(self, file_path: str) -> bool:
        """Checks if the loader can handle the given file path."""
        if not os.path.isfile(file_path):
            return False
        file_type = get_file_extension(file_path)
        return file_type.lower() in [fmt.lower() for fmt in self.SUPPORTED_FORMATS]

    def _can_handle_stream(self, stream: BytesIO) -> bool:
        """Checks if the loader can handle the given BytesIO stream."""
        try:
            stream.seek(0)
            img = Image.open(stream)
            file_type = img.format.lower()
            stream.seek(0)
            return file_type.lower() in [fmt.lower() for fmt in self.SUPPORTED_FORMATS]
        except Exception:
            return False
                
    @abstractmethod
    def load_content_from_file(self, file_path: str) -> Union[str, object]:
        pass

    @abstractmethod
    def load_content_from_stream(self, stream: BytesIO) -> Union[str, object]:
        pass

    def load(self, source: Union[str, BytesIO]) -> Any:
        if not self.can_handle(source):
            raise ValueError("Unsupported file type or stream.")
        if isinstance(source, str):
            return self.load_content_from_file(source)
        elif isinstance(source, BytesIO):
            return self.load_content_from_stream(source)
        else:
            raise ValueError("Source must be a file path or a stream.")

    def getContent(self) -> Any:
        return self.content

    def load_content_list(self, input_data: Union[str, BytesIO, List[Union[str, BytesIO]]]) -> Union[str, List[str]]:
        if isinstance(input_data, (str, BytesIO)):
            return self.load_content_from_stream_list(input_data)
        elif isinstance(input_data, list):
            return self.load_content_from_file_list(input_data)
        else:
            raise Exception(f"Unsupported input type: {type(input_data)}")

    @abstractmethod
    def load_content_from_stream_list(self, stream: BytesIO) -> List[Any]:
        pass

    @abstractmethod
    def load_content_from_file_list(self, file_path: str) -> List[Any]:
        pass

    def convert_to_images(self, file: Union[str, io.BytesIO], scale: float = 300 / 72) -> Dict[int, bytes]:
        if not self.can_handle(file):
            raise ValueError("Unsupported file type or stream.")
        # Determine if the input is a file path or a stream
        if isinstance(file, str):
            return self._convert_file_to_images(file, scale)
        elif isinstance(file, io.BytesIO):
            return self._convert_stream_to_images(file, scale)
        else:
            raise TypeError("file must be a file path (str) or a BytesIO stream")

    def _convert_file_to_images(self, file_path: str, scale: float) -> Dict[int, bytes]:
        # Check if the file is already an image
        try:
            Image.open(file_path)
            is_image = True
        except IOError:
            is_image = False

        if is_image:
            # If it is, return it as is
            with open(file_path, "rb") as f:
                return {0: f.read()}

        # If it's not an image, proceed with the conversion
        return self._convert_pdf_to_images(pdfium.PdfDocument(file_path), scale)

    def _convert_stream_to_images(self, file_stream: io.BytesIO, scale: float) -> Dict[int, bytes]:
        # Check if the stream is already an image
        try:
            Image.open(file_stream)
            is_image = True
        except IOError:
            is_image = False

        # Reset stream position
        file_stream.seek(0)

        if is_image:
            # If it is, return it as is
            return {0: file_stream.read()}

        # If it's not an image, proceed with the conversion
        # Note: pdfium.PdfDocument may not support streams directly.
        # You might need to save the stream to a temporary file first.
        return self._convert_pdf_to_images(pdfium.PdfDocument(file_stream), scale)

    def _convert_pdf_to_images(self, pdf_file, scale: float) -> Dict[int, bytes]:
        # Get all pages at once
        renderer = pdf_file.render(
            pdfium.PdfBitmap.to_pil,
            page_indices=list(range(len(pdf_file))),
            scale=scale,
        )
        
        # Convert all images to bytes and store in dictionary
        final_images = {}
        for page_index, image in enumerate(renderer):
            image_byte_array = BytesIO()
            image.save(image_byte_array, format="jpeg", optimize=True)
            final_images[page_index] = image_byte_array.getvalue()
            
        return final_images

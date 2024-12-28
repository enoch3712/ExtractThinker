from abc import ABC, abstractmethod
import io
from PIL import Image
from io import BytesIO
import pypdfium2 as pdfium
from typing import Any, Dict, Union
from cachetools import TTLCache
import os
import magic
from extract_thinker.utils import get_file_extension, check_mime_type

class DocumentLoader(ABC):
    def __init__(self, content: Any = None, cache_ttl: int = 300):
        self.content = content
        self.file_path = None
        self.cache = TTLCache(maxsize=100, ttl=cache_ttl)
        self.vision_mode = False

    def set_vision_mode(self, enabled: bool = True) -> None:
        """Enable or disable vision mode processing."""
        self.vision_mode = enabled

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
            # Read the first few bytes to determine file type
            mime = magic.from_buffer(stream.getvalue(), mime=True)
            stream.seek(0)  # Reset stream position
            return check_mime_type(mime, self.SUPPORTED_FORMATS)
        except Exception:
            return False
                
    @abstractmethod
    def load(self, source: Union[str, BytesIO]) -> Any:
        """Enhanced load method that handles vision mode."""
        pass

    def getContent(self) -> Any:
        return self.content

    def convert_to_images(self, file: Union[str, io.BytesIO, io.BufferedReader], scale: float = 300 / 72) -> Dict[int, bytes]:
        # Determine if the input is a file path or a stream
        if isinstance(file, str):
            return self._convert_file_to_images(file, scale)
        elif isinstance(file, (io.BytesIO, io.BufferedReader)):  # Accept both BytesIO and BufferedReader
            return self._convert_stream_to_images(file, scale)
        else:
            raise TypeError("file must be a file path (str) or a file-like stream")

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

    def can_handle_vision(self, source: Union[str, BytesIO]) -> bool:
        """
        Checks if the loader can handle the source in vision mode.
        
        Args:
            source: Either a file path (str) or a BytesIO stream
            
        Returns:
            bool: True if the loader can handle the source in vision mode
        """
        try:
            if isinstance(source, str):
                ext = get_file_extension(source).lower()
                return ext in ['pdf', 'jpg', 'jpeg', 'png', 'tiff', 'bmp']
            elif isinstance(source, BytesIO):
                try:
                    Image.open(source)
                    return True
                except:
                    # Try to load as PDF
                    try:
                        pdfium.PdfDocument(source)
                        return True
                    except:
                        return False
            return False
        except Exception:
            return False

    def can_handle_paginate(self, source: Union[str, BytesIO]) -> bool:
        """
        Checks if the source supports pagination (e.g., PDF, PPT).
        
        Args:
            source: Either a file path (str) or a BytesIO stream
            
        Returns:
            bool: True if the source supports pagination
        """
        try:
            if isinstance(source, str):
                # For file paths, check the extension
                ext = get_file_extension(source).lower()
            else:
                # For BytesIO streams, use magic to detect mime type
                mime = magic.from_buffer(source.getvalue(), mime=True)
                source.seek(0)  # Reset stream position
                return mime == 'application/pdf'

            # List of extensions that support pagination
            return ext in ['pdf']
        except Exception:
            return False  
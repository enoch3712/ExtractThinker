from abc import ABC, abstractmethod
import io
from io import BytesIO
from PIL import Image
import pypdfium2 as pdfium
from typing import Any, Dict, Union, List
from cachetools import TTLCache
import os
import magic
from extract_thinker.utils import get_file_extension, check_mime_type
from playwright.sync_api import sync_playwright
from urllib.parse import urlparse
import base64
import math

class DocumentLoader(ABC):
    # SUPPORTED_FORMATS = [
    #     "pdf", "jpg", "jpeg", "png", "tiff", "bmp"
    # ]

    def __init__(self, content: Any = None, cache_ttl: int = 300):
        """Initialize loader.
        
        Args:
            content: Initial content
            cache_ttl: Cache time-to-live in seconds
        """
        self.content = content
        self.file_path = None
        self.cache = TTLCache(maxsize=100, ttl=cache_ttl)
        self.vision_mode = False
        self.max_image_size = None  # Changed to None by default

    def set_max_image_size(self, size: int) -> None:
        """Set the maximum image size."""
        self.max_image_size = size

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
        """Convert file to images, handling both URLs and local files."""
        # Check if it's a URL
        if self.is_url(file_path):
            try:
                screenshot = self._capture_screenshot_from_url(file_path)
                # Convert screenshot to PIL Image for potential resizing
                img = Image.open(BytesIO(screenshot))
                img = self._resize_if_needed(img)
                
                # Split into vertical chunks
                chunks = self._split_image_vertically(img)
                
                # Return dictionary with chunks as list
                return {0: chunks}  # All chunks from URL are considered "page 0"
                
            except Exception as e:
                raise ValueError(f"Failed to capture screenshot from URL: {str(e)}")
        
        # Existing code for local files...
        try:
            Image.open(file_path)
            is_image = True
        except IOError:
            is_image = False

        if is_image:
            with open(file_path, "rb") as f:
                return {0: f.read()}

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
        return self._convert_pdf_to_images(pdfium.PdfDocument(file_stream), scale)

    def _resize_if_needed(self, image: Image.Image) -> Image.Image:
        """Resize image if it exceeds maximum dimensions while maintaining aspect ratio.
        
        Args:
            image: PIL Image object
            
        Returns:
            PIL Image object (resized if necessary)
        """
        if self.max_image_size is None:  # Skip resizing if max_image_size not set
            return image
            
        width, height = image.size
        if width > self.max_image_size or height > self.max_image_size:
            # Calculate scaling factor to fit within max dimensions
            scale = self.max_image_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return image

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
            # Resize image if needed
            image = self._resize_if_needed(image)
            image_byte_array = BytesIO()
            image.save(image_byte_array, format="jpeg", optimize=True)
            final_images[page_index] = image_byte_array.getvalue()
            
        return final_images

    def can_handle_vision(self, source: Union[str, BytesIO]) -> bool:
        """
        Checks if the loader can handle the source in vision mode.
        
        Args:
            source: Either a file path (str), URL, or a BytesIO stream
            
        Returns:
            bool: True if the loader can handle the source in vision mode
        """
        try:
            if isinstance(source, str):
                if self.is_url(source):
                    return True  # URLs are always supported in vision mode
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

    def is_url(self, source: str) -> bool:
        """Check if the source is a URL."""
        try:
            result = urlparse(source)
            return bool(result.scheme and result.netloc)
        except:
            return False

    def _capture_screenshot_from_url(self, url: str) -> bytes:
        """
        Captures a full-page screenshot of a URL using Playwright.
        
        Args:
            url: The URL to capture
            
        Returns:
            bytes: The screenshot image data
        """
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            try:
                # Navigate to URL
                page.goto(url, wait_until='networkidle')
                
                # Optional: Handle cookie consent popups (customize selectors as needed)
                try:
                    page.click('button:has-text("Accept")', timeout=3000)
                except:
                    pass  # Ignore if no cookie banner found
                    
                # Wait for content to load
                page.wait_for_timeout(1000)
                
                # Capture full page screenshot
                screenshot = page.screenshot(full_page=True)
                
                return screenshot
                
            finally:
                browser.close()

    def _split_image_vertically(self, img: Image.Image, chunk_height: int = 1000) -> List[bytes]:
        """
        Splits a tall PIL Image into vertical chunks of `chunk_height`.
        Returns a list of bytes in PNG format, in top-to-bottom order.
        
        Args:
            img: PIL Image to split
            chunk_height: Height of each chunk in pixels
            
        Returns:
            List of PNG-encoded bytes for each chunk
        """
        width, height = img.size
        num_chunks = math.ceil(height / chunk_height)

        chunks_bytes = []
        for i in range(num_chunks):
            top = i * chunk_height
            bottom = min((i + 1) * chunk_height, height)
            crop_box = (0, top, width, bottom)
            
            # Crop the chunk
            chunk_img = img.crop(crop_box)
            
            # Convert chunk to bytes
            chunk_bytes = io.BytesIO()
            chunk_img.save(chunk_bytes, format="PNG", optimize=True)
            chunk_bytes.seek(0)
            chunks_bytes.append(chunk_bytes.read())
            
        return chunks_bytes

def is_url(source: str) -> bool:
    """Check if the source is a URL."""
    try:
        result = urlparse(source)
        return bool(result.scheme and result.netloc)
    except:
        return False  
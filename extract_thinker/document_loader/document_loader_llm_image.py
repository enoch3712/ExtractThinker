import io
from typing import Any, Dict, List, Union, Optional
from io import BytesIO
from operator import attrgetter
from cachetools import cachedmethod
from cachetools.keys import hashkey
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader
from dataclasses import dataclass, field
import os


@dataclass
class LLMImageConfig:
    """Configuration for LLM Image loader.
    
    Args:
        content: Initial content (optional)
        cache_ttl: Cache time-to-live in seconds (default: 300)
        llm: Language model to use for image processing (optional)
        max_image_size: Maximum image size in bytes (default: None)
        image_format: Format to convert images to (default: None)
        compression_quality: Image compression quality (default: 85)
    """
    # Optional parameters
    content: Optional[Any] = None
    cache_ttl: int = 300
    llm: Optional[Any] = None
    max_image_size: Optional[int] = None
    image_format: Optional[str] = None
    compression_quality: int = 85

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.compression_quality < 0 or self.compression_quality > 100:
            raise ValueError("compression_quality must be between 0 and 100")
        
        if self.max_image_size is not None and self.max_image_size <= 0:
            raise ValueError("max_image_size must be positive")
        
        if self.image_format is not None and self.image_format.lower() not in ['jpeg', 'png', 'webp']:
            raise ValueError("image_format must be one of: jpeg, png, webp")


class DocumentLoaderLLMImage(CachedDocumentLoader):
    """
    Document loader that handles images and PDFs, converting them to a format suitable for vision LLMs.
    This loader is used as a fallback when no other loader is available and vision mode is required.
    """
    SUPPORTED_FORMATS = ['pdf', 'jpg', 'jpeg', 'png', 'tiff', 'bmp']
    
    def __init__(
        self,
        content_or_config: Union[Any, LLMImageConfig] = None,
        cache_ttl: int = 300,
        llm: Optional[Any] = None,
        max_image_size: Optional[int] = None,
        image_format: Optional[str] = None,
        compression_quality: int = 85
    ):
        """Initialize loader.
        
        Args:
            content_or_config: Either a LLMImageConfig object or initial content
            cache_ttl: Cache time-to-live in seconds (only used if content_or_config is not LLMImageConfig)
            llm: Language model to use for image processing (only used if content_or_config is not LLMImageConfig)
            max_image_size: Maximum image size in bytes (only used if content_or_config is not LLMImageConfig)
            image_format: Format to convert images to (only used if content_or_config is not LLMImageConfig)
            compression_quality: Image compression quality (only used if content_or_config is not LLMImageConfig)
        """
        # Handle both config-based and old-style initialization
        if isinstance(content_or_config, LLMImageConfig):
            self.config = content_or_config
        else:
            # Create config from individual parameters
            self.config = LLMImageConfig(
                content=content_or_config,
                cache_ttl=cache_ttl,
                llm=llm,
                max_image_size=max_image_size,
                image_format=image_format,
                compression_quality=compression_quality
            )
        
        super().__init__(self.config.content, self.config.cache_ttl)
        self.llm = self.config.llm
        self.vision_mode = True  # Always in vision mode since this is for image processing

    def _process_image(self, image_bytes: bytes) -> bytes:
        """Process image according to configuration."""
        if not any([self.config.max_image_size, self.config.image_format, self.config.compression_quality != 85]):
            return image_bytes

        try:
            from PIL import Image
            image = Image.open(BytesIO(image_bytes))

            # Convert format if specified
            if self.config.image_format:
                if image.format != self.config.image_format.upper():
                    # Convert to RGB if saving as JPEG
                    if self.config.image_format.lower() == 'jpeg' and image.mode in ('RGBA', 'P'):
                        image = image.convert('RGB')
                    
                    output = BytesIO()
                    image.save(output, format=self.config.image_format.upper(), 
                             quality=self.config.compression_quality)
                    image_bytes = output.getvalue()

            # Check size limit and apply compression if needed
            if self.config.max_image_size and len(image_bytes) > self.config.max_image_size:
                # First try quality reduction
                quality = self.config.compression_quality
                while len(image_bytes) > self.config.max_image_size and quality > 5:
                    quality -= 5
                    output = BytesIO()
                    if image.mode in ('RGBA', 'P'):
                        image = image.convert('RGB')
                    image.save(output, format='JPEG', quality=quality)
                    image_bytes = output.getvalue()

                # If quality reduction alone didn't work, try resizing
                if len(image_bytes) > self.config.max_image_size:
                    ratio = (self.config.max_image_size / len(image_bytes)) ** 0.5
                    new_size = tuple(int(dim * ratio) for dim in image.size)
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                    output = BytesIO()
                    image.save(output, format='JPEG', quality=quality)
                    image_bytes = output.getvalue()

            return image_bytes
        except Exception as e:
            # If image processing fails, return original bytes
            return image_bytes

    @cachedmethod(cache=attrgetter('cache'), 
                  key=lambda self, source: hashkey(source if isinstance(source, str) else source.getvalue(), self.vision_mode))
    def load(self, source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        """
        Load the source and convert it to a list of pages with images.
        Each page will be a dictionary with:
        - 'content': Empty string (since this loader doesn't extract text)
        - 'image': The image bytes for that page
        
        Args:
            source: Either a file path or a BytesIO stream
            
        Returns:
            List[Dict[str, Any]]: List of pages, each with 'content' and 'image' keys
        """
        if not self.can_handle(source):
            raise ValueError(f"Cannot handle source: {source}")

        try:
            # Convert source to images using the base class's convert_to_images method
            images_dict = self.convert_to_images(source)
            
            # Convert to our standard page-based format
            pages = []
            for page_idx, image_bytes in images_dict.items():
                # Process image according to configuration
                processed_bytes = self._process_image(image_bytes)
                
                pages.append({
                    "content": "",  # No text content since this is image-only
                    "image": processed_bytes
                })
            
            return pages
            
        except Exception as e:
            raise ValueError(f"Failed to load image content: {str(e)}")

    def can_handle(self, source: Union[str, BytesIO]) -> bool:
        """Check if this loader can handle the source."""
        if isinstance(source, BytesIO):
            try:
                # Try to open as image to verify it's a valid image file
                from PIL import Image
                source.seek(0)
                Image.open(source)
                source.seek(0)
                return True
            except Exception:
                return False
        elif isinstance(source, str):
            try:
                # Check if it's a valid file path
                if not os.path.exists(source):
                    return False
                # Check extension
                ext = os.path.splitext(source)[1].lower().lstrip('.')
                return ext in self.SUPPORTED_FORMATS
            except Exception:
                return False
        return False

    def can_handle_vision(self, source: Union[str, BytesIO]) -> bool:
        """
        Check if this loader can handle the source in vision mode.
        This loader is specifically for vision/image processing.
        """
        # This loader is always in vision mode and can handle any supported format
        return True
import io
import os
import json
import base64
import requests
from typing import Any, Union, Dict, List, Optional
from io import BytesIO
from operator import attrgetter
from cachetools import cachedmethod
from cachetools.keys import hashkey
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader
from extract_thinker.utils import get_file_extension
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass
class MistralOCRConfig:
    """Configuration for Mistral OCR loader.
    
    Args:
        api_key: Mistral API key
        model: OCR model to use
        content: Initial content (optional)
        cache_ttl: Cache time-to-live in seconds (default: 300)
        include_image_base64: Whether to include image base64 in response (default: False)
        pages: Specific pages to process (optional)
        image_limit: Maximum number of images to extract (optional)
        image_min_size: Minimum image size to extract (optional)
    """
    api_key: str
    model: str = "mistral-ocr-latest"
    content: Optional[Any] = None
    cache_ttl: int = 300
    include_image_base64: bool = False
    pages: Optional[List[int]] = None
    image_limit: Optional[int] = None
    image_min_size: Optional[int] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.api_key:
            raise ValueError("api_key is required")
        
        if self.cache_ttl <= 0:
            raise ValueError("cache_ttl must be positive")


class DocumentLoaderMistralOCR(CachedDocumentLoader):
    """Loader for documents using Mistral OCR API to extract text and images."""
    SUPPORTED_FORMATS = ['pdf', 'jpg', 'jpeg', 'png', 'tiff', 'bmp']
    
    def __init__(
        self,
        config: MistralOCRConfig
    ):
        """Initialize loader.
        
        Args:
            config: MistralOCRConfig object with required configuration
        """
        self.config = config
        super().__init__(self.config.content, self.config.cache_ttl)
        self.vision_mode = True  # OCR is always vision-enabled
        self.api_url = "https://api.mistral.ai/v1/ocr"

    def _is_url(self, source: str) -> bool:
        """Check if the source string is a URL."""
        try:
            result = urlparse(source)
            return bool(result.scheme and result.netloc)
        except:
            return False

    def _prepare_payload(self, source: Union[str, BytesIO]) -> Dict:
        """Prepare payload for Mistral OCR API request."""
        payload = {
            "model": self.config.model,
            "include_image_base64": self.config.include_image_base64
        }

        # Handle pages if specified
        if self.config.pages:
            payload["pages"] = self.config.pages

        # Handle image limit if specified
        if self.config.image_limit is not None:
            payload["image_limit"] = self.config.image_limit

        # Handle image minimum size if specified
        if self.config.image_min_size is not None:
            payload["image_min_size"] = self.config.image_min_size

        # Handle source (file path, URL, or BytesIO)
        if isinstance(source, str):
            if self._is_url(source):
                # Direct URL
                payload["document"] = {
                    "type": "document_url",
                    "document_url": source
                }
            else:
                # File path - need to upload to Mistral
                file_id = self._upload_file_to_mistral(source)
                signed_url = self._get_signed_url(file_id)
                payload["document"] = {
                    "type": "document_url",
                    "document_url": signed_url
                }
        else:
            # BytesIO object - need to upload to Mistral
            file_id = self._upload_file_to_mistral(source)
            signed_url = self._get_signed_url(file_id)
            payload["document"] = {
                "type": "document_url",
                "document_url": signed_url
            }

        return payload

    def _upload_file_to_mistral(self, source: Union[str, BytesIO]) -> str:
        """Upload a file to Mistral's file storage system.
        
        Args:
            source: Either a file path or BytesIO object
            
        Returns:
            str: The file ID returned by Mistral
        """
        upload_url = "https://api.mistral.ai/v1/files"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}"
        }
        
        files = None
        
        try:
            if isinstance(source, str):
                # File path
                with open(source, 'rb') as file:
                    files = {
                        'file': (os.path.basename(source), file, 'application/octet-stream'),
                        'purpose': (None, 'ocr')
                    }
                    response = requests.post(upload_url, headers=headers, files=files)
            else:
                # BytesIO object
                source.seek(0)
                files = {
                    'file': ('document.bin', source, 'application/octet-stream'),
                    'purpose': (None, 'ocr')
                }
                response = requests.post(upload_url, headers=headers, files=files)
            
            response.raise_for_status()
            result = response.json()
            return result.get("id")
            
        except requests.exceptions.RequestException as e:
            if hasattr(e.response, 'text'):
                error_text = e.response.text
                try:
                    error_json = json.loads(error_text)
                    if 'error' in error_json:
                        raise ValueError(f"Mistral API upload error: {error_json['error']}")
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"Error uploading file to Mistral: {str(e)}")

    def _get_signed_url(self, file_id: str) -> str:
        """Get a signed URL for an uploaded file.
        
        Args:
            file_id: The file ID returned by Mistral
            
        Returns:
            str: The signed URL
        """
        url = f"https://api.mistral.ai/v1/files/{file_id}/url"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            result = response.json()
            return result.get("url")
            
        except requests.exceptions.RequestException as e:
            if hasattr(e.response, 'text'):
                error_text = e.response.text
                try:
                    error_json = json.loads(error_text)
                    if 'error' in error_json:
                        raise ValueError(f"Mistral API signed URL error: {error_json['error']}")
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"Error getting signed URL from Mistral: {str(e)}")

    @cachedmethod(cache=attrgetter('cache'), 
                  key=lambda self, source: hashkey(source if isinstance(source, str) else source.getvalue()))
    def load(self, source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        """
        Load document from a file path, URL, or BytesIO stream using Mistral OCR API.
        Return a list of dictionaries, each representing one page:
          - "content": extracted text (markdown format)
          - "images": list of image data if available and requested
        """
        if not self.can_handle(source):
            raise ValueError(f"Cannot handle source: {source}")

        # Prepare headers and payload
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }
        
        payload = self._prepare_payload(source)
        
        # Make API request
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                data=json.dumps(payload)
            )
            
            # Check for successful response
            response.raise_for_status()
            result = response.json()
            
            # Process and return the results
            pages_data = []
            for page in result.get("pages", []):
                page_dict = {
                    "content": page.get("markdown", ""),
                    "page_index": page.get("index", 0)
                }
                
                # Include images if available
                if "images" in page and page["images"]:
                    page_dict["images"] = page["images"]
                
                # Include dimensions if available
                if "dimensions" in page:
                    page_dict["dimensions"] = page["dimensions"]
                
                pages_data.append(page_dict)
            
            return pages_data
            
        except requests.exceptions.RequestException as e:
            if hasattr(e.response, 'text'):
                error_text = e.response.text
                try:
                    error_json = json.loads(error_text)
                    if 'error' in error_json:
                        raise ValueError(f"Mistral OCR API error: {error_json['error']}")
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"Error calling Mistral OCR API: {str(e)}")

    def can_handle(self, source: Union[str, BytesIO]) -> bool:
        """
        Checks if the loader can handle the given source.
        For Mistral OCR, we support various document formats and URLs.
        """
        if isinstance(source, str):
            # Handle URLs
            if self._is_url(source):
                return True
            
            # Handle file paths
            if not os.path.isfile(source):
                return False
                
            file_type = get_file_extension(source)
            return file_type.lower() in [fmt.lower() for fmt in self.SUPPORTED_FORMATS]
        
        elif isinstance(source, BytesIO):
            # BytesIO objects are always accepted as we'll send the raw bytes to the API
            return True
            
        return False

    def can_handle_vision(self, source: Union[str, BytesIO]) -> bool:
        """Mistral OCR always supports vision mode."""
        return self.can_handle(source)

    def can_handle_paginate(self, source: Union[str, BytesIO]) -> bool:
        """Check if pagination is supported for this source."""
        # Only PDF documents support pagination
        if isinstance(source, str) and not self._is_url(source):
            ext = get_file_extension(source).lower()
            return ext == 'pdf'
        return False  # Conservative default for other types 
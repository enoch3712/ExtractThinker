import io
import os
import json
import base64
import tempfile
import requests
from typing import Any, Union, Dict, List, Optional
from io import BytesIO
from operator import attrgetter
from cachetools import cachedmethod
from cachetools.keys import hashkey
from concurrent.futures import ThreadPoolExecutor, as_completed
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader
from extract_thinker.utils import get_file_extension
from dataclasses import dataclass
from urllib.parse import urlparse
import re
from PIL import Image


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
        allow_image_recursive: Whether to allow recursive image extraction (default: False)
    """
    api_key: str
    model: str = "mistral-ocr-latest"
    content: Optional[Any] = None
    cache_ttl: int = 300
    include_image_base64: bool = False
    pages: Optional[List[int]] = None
    image_limit: Optional[int] = None
    image_min_size: Optional[int] = None
    allow_image_recursive: bool = False
    
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

    def _get_file_content_type(self, source: str) -> str:
        """Get the content type based on file extension."""
        ext = get_file_extension(source).lower()
        content_type_map = {
            'pdf': 'application/pdf',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'tiff': 'image/tiff',
            'bmp': 'image/bmp'
        }
        return content_type_map.get(ext, 'application/octet-stream')

    def _is_image_file(self, source: Union[str, BytesIO]) -> bool:
        """Check if the source is an image file."""
        if isinstance(source, str):
            ext = get_file_extension(source).lower()
            return ext in ['jpg', 'jpeg', 'png', 'tiff', 'bmp']
        return False

    def _convert_image_to_pdf(self, source: Union[str, BytesIO]) -> BytesIO:
        """Convert an image file to PDF format, which is supported by Mistral OCR API.
        
        Args:
            source: Either a file path or BytesIO object
            
        Returns:
            BytesIO: PDF content
        """
        try:
            from PIL import Image
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            
            # Open the image
            if isinstance(source, str):
                img = Image.open(source)
            else:
                source.seek(0)
                img = Image.open(source)
            
            # Create a BytesIO buffer for the PDF
            pdf_buffer = BytesIO()
            
            # Calculate dimensions to fit on a letter page
            img_width, img_height = img.size
            width, height = letter
            
            # Maintain aspect ratio
            ratio = min(width / img_width, height / img_height) * 0.9  # 90% of the page
            new_width = img_width * ratio
            new_height = img_height * ratio
            
            # Create a temporary file for the image
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img_file:
                img.save(temp_img_file.name, format='PNG')
                
                # Create PDF
                c = canvas.Canvas(pdf_buffer, pagesize=letter)
                # Center the image
                x_centered = (width - new_width) / 2
                y_centered = (height - new_height) / 2
                c.drawImage(temp_img_file.name, x_centered, y_centered, width=new_width, height=new_height)
                c.save()
                
                # Remove temporary file
                os.unlink(temp_img_file.name)
            
            pdf_buffer.seek(0)
            return pdf_buffer
            
        except ImportError:
            raise ImportError("PIL and reportlab are required for image to PDF conversion. Install with: pip install pillow reportlab")

    def _prepare_payload(self, source: Union[str, BytesIO]) -> Dict:
        """Prepare payload for Mistral OCR API request."""
        payload = {
            "model": self.config.model,
            "include_image_base64": self.config.include_image_base64
        }

        # Handle pages if specified (only for PDFs)
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
                # If it's an image file, convert to PDF first
                if self._is_image_file(source):
                    pdf_buffer = self._convert_image_to_pdf(source)
                    file_id = self._upload_file_to_mistral(pdf_buffer)
                    signed_url = self._get_signed_url(file_id)
                    payload["document"] = {
                        "type": "document_url",
                        "document_url": signed_url
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
                content_type = self._get_file_content_type(source)
                
                # If it's a JPEG file, convert to PNG
                if content_type == 'image/jpeg':
                    try:
                        from PIL import Image
                        print(f"Converting JPEG to PNG for file: {source}")
                        img = Image.open(source)
                        # Create a temporary PNG file
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_png:
                            temp_path = temp_png.name
                            img.save(temp_path, format='PNG')
                            
                        # Use this PNG file instead
                        with open(temp_path, 'rb') as file:
                            files = {
                                'file': (os.path.basename(temp_path), file, 'image/png'),
                                'purpose': (None, 'ocr')
                            }
                            print(f"Uploading converted PNG file with content type image/png")
                            response = requests.post(upload_url, headers=headers, files=files)
                        
                        # Delete temporary file
                        os.unlink(temp_path)
                    except ImportError:
                        raise ImportError("PIL is required for image conversion. Install with: pip install pillow")
                else:
                    # Not a JPEG, upload normally
                    with open(source, 'rb') as file:
                        files = {
                            'file': (os.path.basename(source), file, content_type),
                            'purpose': (None, 'ocr')
                        }
                        print(f"Uploading file {source} with content type {content_type}")
                        response = requests.post(upload_url, headers=headers, files=files)
            else:
                # BytesIO object
                source.seek(0)
                # For BytesIO objects, we need to determine content type
                content_type = 'application/octet-stream'
                
                # Try to detect content type from first few bytes
                source.seek(0)
                header = source.read(8)
                source.seek(0)
                
                # Check for common file signatures
                if header.startswith(b'\x89PNG\r\n\x1a\n'):
                    content_type = 'image/png'
                    filename = 'image.png'
                elif header.startswith(b'\xff\xd8'):
                    # It's a JPEG, we need to convert to PNG
                    try:
                        from PIL import Image
                        print("Converting JPEG to PNG from BytesIO")
                        source.seek(0)
                        img = Image.open(source)
                        # Create a BytesIO for the PNG
                        png_buffer = BytesIO()
                        img.save(png_buffer, format='PNG')
                        png_buffer.seek(0)
                        
                        # Use this PNG buffer instead
                        content_type = 'image/png'
                        filename = 'image.png'
                        source = png_buffer
                    except ImportError:
                        raise ImportError("PIL is required for image conversion. Install with: pip install pillow")
                elif header.startswith(b'%PDF'):
                    content_type = 'application/pdf'
                    filename = 'document.pdf'
                elif header.startswith(b'GIF87a') or header.startswith(b'GIF89a'):
                    content_type = 'image/gif'
                    filename = 'image.gif'
                else:
                    # Default to PDF for unknown types
                    content_type = 'application/pdf'
                    filename = 'document.pdf'
                
                print(f"Uploading BytesIO with content type: {content_type}")
                
                files = {
                    'file': (filename, source, content_type),
                    'purpose': (None, 'ocr')
                }
                response = requests.post(upload_url, headers=headers, files=files)
            
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                error_detail = "Unknown error"
                try:
                    error_json = response.json()
                    if "error" in error_json:
                        error_detail = error_json.get("error", {})
                        if isinstance(error_detail, dict):
                            error_detail = error_detail.get("message", str(error_detail))
                except:
                    error_detail = response.text if response.text else str(e)
                
                print(f"Error uploading file to Mistral: {error_detail}")
                print(f"Status code: {response.status_code}")
                print(f"Request URL: {upload_url}")
                print(f"Content type: {content_type}")
                raise ValueError(f"Mistral API upload error: {error_detail}")
                
            result = response.json()
            file_id = result.get("id")
            if not file_id:
                raise ValueError(f"No file ID returned from Mistral API: {result}")
                
            print(f"Successfully uploaded file to Mistral with ID: {file_id}")
            return file_id
            
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                error_text = e.response.text
                try:
                    error_json = json.loads(error_text)
                    if 'error' in error_json:
                        error_message = error_json.get('error', {})
                        if isinstance(error_message, dict) and 'message' in error_message:
                            error_message = error_message.get('message')
                        raise ValueError(f"Mistral API upload error: {error_message}")
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
            if hasattr(e, 'response') and e.response is not None:
                error_text = e.response.text
                try:
                    error_json = json.loads(error_text)
                    if 'error' in error_json:
                        error_message = error_json.get('error', {})
                        if isinstance(error_message, dict) and 'message' in error_message:
                            error_message = error_message.get('message')
                        raise ValueError(f"Mistral API signed URL error: {error_message}")
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"Error getting signed URL from Mistral: {str(e)}")

    def _process_image_recursively(self, image_data: bytes, original_img_id: str) -> Optional[Dict[str, Any]]:
        """
        Processes image bytes by converting to PNG, uploading it as a NEW file,
        getting a NEW signed URL, and calling Mistral OCR via that new URL.
        Returns extracted text or a placeholder if extraction fails. NO LLM FALLBACK.

        Args:
            image_data: The image data in bytes format.
            original_img_id: The original ID (e.g., "img-0.jpeg") for logging.

        Returns:
            Dict[str, Any]: Dict with 'content' and 'page_index'.
                            'content' contains extracted text or a placeholder message.
        """
        # Default failure content
        failure_content = f"[Image processing failed for {original_img_id}]"
        # Default return structure
        result_dict = {"content": failure_content, "page_index": 0}

        if Image is None:
             print(f"Error processing {original_img_id}: Pillow (PIL) not installed.")
             result_dict["content"] = "[Image processing failed: Pillow not installed]"
             return result_dict

        png_buffer = None
        new_file_id = None
        new_signed_url = None
        try:
            # --- Aggressive PNG Conversion ---
            try:
                image_buffer = BytesIO(image_data)
                img = Image.open(image_buffer)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                png_buffer = BytesIO()
                img.save(png_buffer, format="PNG")
                png_buffer.seek(0)
                # print(f"Converted image {original_img_id} to standard RGB PNG format.") # Optional log
            except Exception as format_err:
                print(f"Error converting {original_img_id} to PNG: {format_err}.")
                result_dict["content"] = f"[Image processing failed: PNG conversion error {format_err}]"
                return result_dict # Return immediately on conversion failure

            # --- Upload PNG as a New File ---
            try:
                 # print(f"Uploading converted PNG data for {original_img_id} as new file...") # Optional log
                 png_buffer.seek(0)
                 new_file_id = self._upload_file_to_mistral(png_buffer)
                 if not new_file_id: raise ValueError("Upload did not return a file ID.")
                 # print(f"Successfully uploaded {original_img_id} as new file ID: {new_file_id}") # Optional log
            except Exception as upload_err:
                 print(f"Error uploading new PNG file for {original_img_id}: {upload_err}")
                 result_dict["content"] = f"[Image processing failed: Upload error {upload_err}]"
                 return result_dict # Return on upload failure

            # --- Get New Signed URL ---
            try:
                 new_signed_url = self._get_signed_url(new_file_id)
                 if not new_signed_url: raise ValueError("Could not get signed URL.")
                 # print(f"Got new signed URL for file {new_file_id}") # Optional log
            except Exception as url_err:
                 print(f"Error getting signed URL for new file {new_file_id}: {url_err}")
                 result_dict["content"] = f"[Image processing failed: Signed URL error {url_err}]"
                 return result_dict # Return on URL failure

            # --- Prepare API Call with New Signed URL ---
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}"
            }
            payload = {
                "model": self.config.model,
                "include_image_base64": False,
                "document": {
                    "type": "image_url",
                    "image_url": new_signed_url
                }
            }

            # --- Make the API Call ---
            # print(f"Attempting OCR for {original_img_id} via new signed URL ({new_file_id})...") # Optional log
            response = requests.post(
                self.api_url,
                headers=headers,
                data=json.dumps(payload)
            )

            # --- Process Response ---
            try:
                response.raise_for_status()
                result = response.json()
                # print(f"OCR request for new file {new_file_id} successful (HTTP Status).") # Optional log

                if result.get("pages") and len(result["pages"]) > 0:
                    page = result["pages"][0]
                    extracted_markdown = page.get("markdown", "")

                    # Check if only an image reference was returned
                    img_ref_match = re.match(r"^\s*!\[(.*?)\]\((.*?)\)\s*$", extracted_markdown)
                    if img_ref_match and img_ref_match.group(1) == img_ref_match.group(2):
                         print(f"Warning: OCR via new signed URL for {original_img_id} returned image reference: {extracted_markdown}")
                         hidden_text = page.get("text", "")
                         if hidden_text:
                             print(f"Using hidden 'text' field content for {original_img_id}.")
                             result_dict["content"] = hidden_text
                         else:
                             result_dict["content"] = "[Image content not extracted: API returned reference only]"
                    elif not extracted_markdown:
                         print(f"Warning: OCR for {original_img_id} returned empty markdown.")
                         result_dict["content"] = "[Image content not extracted: Empty markdown response]"
                    else:
                        # Success! Got actual markdown content
                        # print(f"Successfully extracted text for {original_img_id} via new signed URL.") # Optional log
                        result_dict["content"] = extracted_markdown
                else:
                    print(f"Warning: No pages found in Mistral OCR response for new file {new_file_id}.")
                    result_dict["content"] = "[Image content not extracted: No pages in API response]"

            except requests.exceptions.HTTPError as http_err:
                 error_detail = f"HTTP Error {http_err.response.status_code}"
                 try:
                     error_json = http_err.response.json()
                     if 'error' in error_json: msg = error_json['error'].get('message', str(error_json['error']))
                     else: msg = http_err.response.text
                     error_detail += f" - {msg}"
                 except json.JSONDecodeError: error_detail += f" - Body: {http_err.response.text}"
                 print(f"Error during OCR call for new file {new_file_id}: {error_detail}")
                 result_dict["content"] = f"[Image content extraction failed: {error_detail}]"

        except Exception as e:
            # Catch-all for unexpected errors during the setup phase (e.g., Pillow issues)
            print(f"Unexpected error processing image {original_img_id}: {str(e)}")
            # Ensure content reflects the error
            if result_dict["content"] == failure_content: # If no specific error was set yet
                 result_dict["content"] = f"[Image content extraction failed: {str(e)}]"

        return result_dict # Always return the dictionary

    @cachedmethod(cache=attrgetter('cache'), 
                  key=lambda self, source: hashkey(source if isinstance(source, str) else source.getvalue()))
    def load(self, source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        """
        Load document from a file path, URL, or BytesIO stream using Mistral OCR API.
        Return a list of dictionaries, each representing one page:
          - "content": extracted text (markdown format)
          - "images": list of processed image content if available
        """
        if not self.can_handle(source):
            raise ValueError(f"Cannot handle source: {source}")

        # Convert image file to PDF if needed
        original_source = source
        if isinstance(source, str) and not self._is_url(source) and self._is_image_file(source):
            source = self._convert_image_to_pdf(source)

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
                
                # Include dimensions if available
                if "dimensions" in page:
                    page_dict["dimensions"] = page["dimensions"]

                image_extraction_results = {} # Store img_id -> extracted_text
                if "images" in page and page["images"] and self.config.allow_image_recursive:
                    futures = []
                    with ThreadPoolExecutor() as executor:
                        for img in page["images"]:
                            img_id = img.get("id")
                            # Use a distinct variable name and check its type
                            img_base64_str = img.get("image_base64") 
                            
                            # --- Check if base64 data exists and is a string ---
                            if img_id and img_base64_str and isinstance(img_base64_str, str):
                                try:
                                    processed_base64 = img_base64_str # Start with the original string
                                    # Check for data URI prefix ONLY if it's a string
                                    if img_base64_str.startswith("data:"):
                                        processed_base64 = img_base64_str.split(",", 1)[1]
                                    
                                    # Ensure we have data after potential split before decoding
                                    if processed_base64:
                                         image_bytes = base64.b64decode(processed_base64)
                                         # Submit to the function that uploads fresh
                                         future = executor.submit(
                                             self._process_image_recursively, # Call the re-uploading function
                                             image_bytes,
                                             img_id # Pass original ID for logging
                                         )
                                         futures.append((future, img_id))
                                    else:
                                         print(f"Warning: Base64 data for image {img_id} was empty after processing prefix.")

                                except Exception as decode_err:
                                     print(f"Warning: Failed to decode/process base64 for image {img_id}: {decode_err}")
                            elif img_id:
                                 # Log if base64 data is missing or not a string
                                 print(f"Warning: Missing or invalid type for image_base64 data for image {img_id} (Type: {type(img_base64_str)}). Skipping image processing for this item.")
                            # --- End Check ---

                        # Collect results as they complete
                        for future, img_id in futures:
                            try:
                                processed_content = future.result()
                                if processed_content:
                                    image_extraction_results[img_id] = processed_content["content"]
                            except Exception as e:
                                print(f"Warning: Failed to process image {img_id}: {e}")
                    
                    if image_extraction_results:
                        page_dict["images"] = [{"id": img_id, "content": image_extraction_results[img_id]} for img_id in image_extraction_results]
                        
                        # Replace image references in content with extracted text
                        page_content = page_dict["content"]
                        for img_id, extracted_text in image_extraction_results.items():
                            image_marker = f"![{img_id}]({img_id})"
                            if image_marker in page_content:
                                # If replacement text is non-empty, add it with a newline
                                if extracted_text:
                                    replacement = f"\n\n[Image content: {extracted_text}]\n\n"
                                else:
                                    replacement = "\n\n[Image with no extracted text]\n\n"
                                page_content = page_content.replace(image_marker, replacement)
                        
                        page_dict["content"] = page_content

                # Only try to convert images if not a URL - URLs cause issues with Playwright
                if self.vision_mode and not (isinstance(original_source, str) and self._is_url(original_source)):
                    try:
                        images_dict = self.convert_to_images(original_source)
                        if page.get("index") in images_dict:
                            page_dict["image"] = images_dict[page.get("index")]
                    except Exception as img_err:
                        # Log the error but continue without the image
                        print(f"Warning: Could not convert image: {str(img_err)}")
                
                pages_data.append(page_dict)
            
            return pages_data
            
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                error_text = e.response.text
                try:
                    error_json = json.loads(error_text)
                    if 'error' in error_json:
                        error_message = error_json.get('error', {})
                        if isinstance(error_message, dict) and 'message' in error_message:
                            error_message = error_message.get('message')
                        raise ValueError(f"Mistral OCR API error: {error_message}")
                    else:
                        raise ValueError(f"Mistral OCR API error: {error_text}")
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
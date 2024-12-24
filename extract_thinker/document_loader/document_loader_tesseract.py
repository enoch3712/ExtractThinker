from io import BytesIO
import os
from typing import Any, Dict, List, Union
from PIL import Image
import pytesseract
import threading
from queue import Queue
from operator import attrgetter
from cachetools import cachedmethod
from cachetools.keys import hashkey
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader
from extract_thinker.utils import is_pdf_stream


class DocumentLoaderTesseract(CachedDocumentLoader):
    """Document loader for OCR using Tesseract."""
    
    SUPPORTED_FORMATS = ["jpeg", "png", "bmp", "tiff", "pdf", "jpg"]
    
    def __init__(self, tesseract_cmd, isContainer=False, content=None, cache_ttl=300):
        super().__init__(content, cache_ttl)
        self.tesseract_cmd = tesseract_cmd
        if isContainer:
            self.tesseract_cmd = os.environ.get("TESSERACT_PATH", "tesseract")
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        if not os.path.isfile(self.tesseract_cmd):
            raise ValueError(f"Tesseract not found at {self.tesseract_cmd}")

    @cachedmethod(cache=attrgetter('cache'), 
                  key=lambda self, source: hashkey(source if isinstance(source, str) else source.getvalue(), self.vision_mode))
    def load(self, source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        """
        Load and process a document using Tesseract OCR.
        Returns a list of pages, each containing:
        - content: The extracted text content
        - image: The original image (if vision_mode is True)
        
        Args:
            source: Either a file path or BytesIO stream
            
        Returns:
            List[Dict[str, Any]]: List of pages with content and optional images
        """
        if not self.can_handle(source):
            raise ValueError(f"Cannot handle source: {source}")

        try:
            # Process based on source type and format
            if isinstance(source, str):
                if is_pdf_stream(source):
                    with open(source, 'rb') as file:
                        return self._process_pdf(file)
                else:
                    # Single image file
                    image = Image.open(source)
                    return self._process_single_image(image, source)
            else:
                # BytesIO stream
                if is_pdf_stream(source):
                    return self._process_pdf(source)
                else:
                    image = Image.open(source)
                    return self._process_single_image(image, source)

        except Exception as e:
            raise ValueError(f"Error processing document: {str(e)}")

    def _process_pdf(self, pdf_stream: BytesIO) -> List[Dict[str, Any]]:
        """Process a PDF document by converting to images and performing OCR."""
        try:
            # Reset stream position
            pdf_stream.seek(0)
            
            # Convert PDF pages to images
            images_dict = self.convert_to_images(pdf_stream)
            if not images_dict:
                raise ValueError("No images were extracted from PDF")

            # Process each page in parallel
            pages = self._process_images_parallel(images_dict)
            return pages

        except Exception as e:
            raise ValueError(f"Error processing PDF: {str(e)}")

    def _process_single_image(self, image: Image.Image, source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        """Process a single image file."""
        text = str(pytesseract.image_to_string(image))
        page_dict = {
            "content": text
        }

        # If vision mode is enabled, add the original image
        if self.vision_mode:
            if isinstance(source, str):
                with open(source, 'rb') as f:
                    page_dict["image"] = f.read()
            else:
                source.seek(0)
                page_dict["image"] = source.read()

        return [page_dict]

    def _process_images_parallel(self, images_dict: Dict[int, bytes]) -> List[Dict[str, Any]]:
        """Process multiple images in parallel using threads."""
        input_queue = Queue()
        output_queue = Queue()

        # Add all images to the input queue
        for page_num, image_bytes in images_dict.items():
            input_queue.put((page_num, image_bytes))

        # Create worker threads
        threads = []
        num_threads = min(4, len(images_dict))  # Use at most 4 threads
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
        
        # Convert to page dictionaries
        pages = []
        for page_num, text, image_bytes in results:
            page_dict = {
                "content": text
            }
            if self.vision_mode:
                page_dict["image"] = image_bytes
            pages.append(page_dict)

        return pages

    def _worker(self, input_queue: Queue, output_queue: Queue) -> None:
        """Worker thread for parallel image processing."""
        while True:
            item = input_queue.get()
            if item is None:  # Shutdown signal
                break

            try:
                page_num, image_bytes = item
                image_stream = BytesIO(image_bytes)
                image = Image.open(image_stream)
                text = str(pytesseract.image_to_string(image))
                output_queue.put((page_num, text, image_bytes))
            except Exception as e:
                output_queue.put((page_num, str(e), None))
            finally:
                input_queue.task_done()

    def can_handle_vision(self, source: Union[str, BytesIO]) -> bool:
        """Check if this loader can handle the source in vision mode."""
        return self.can_handle(source)

    def load_content_from_file(self, file_path: str) -> Union[str, object]:
        """Legacy method for backward compatibility."""
        pages = self.load(file_path)
        return "\n\n".join(page["content"] for page in pages)

    def load_content_from_stream(self, stream: BytesIO) -> Union[str, object]:
        """Legacy method for backward compatibility."""
        pages = self.load(stream)
        return "\n\n".join(page["content"] for page in pages)

    def load_content_list(self, source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        """Legacy method for backward compatibility."""
        return self.load(source)

    def load_content_from_file_list(self, input: Union[str, List[str]]) -> List[Any]:
        """Legacy method for backward compatibility."""
        if isinstance(input, list):
            all_pages = []
            for file_path in input:
                pages = self.load(file_path)
                all_pages.extend(pages)
            return all_pages
        return self.load(input)

    def load_content_from_stream_list(self, stream: BytesIO) -> List[Any]:
        """Legacy method for backward compatibility."""
        return self.load(stream)

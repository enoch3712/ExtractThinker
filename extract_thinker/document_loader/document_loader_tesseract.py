from io import BytesIO
from operator import attrgetter
import os

import threading
from typing import Any, List, Union
from PIL import Image
import pytesseract

from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader
from extract_thinker.utils import get_file_extension, get_image_type, is_pdf_stream

from cachetools import cachedmethod
from cachetools.keys import hashkey
from queue import Queue

class DocumentLoaderTesseract(CachedDocumentLoader):
    SUPPORTED_FORMATS = ["jpeg", "png", "bmp", "tiff", "pdf"]
    
    def __init__(self, tesseract_cmd, isContainer=False, content=None, cache_ttl=300):
        super().__init__(content, cache_ttl)
        self.tesseract_cmd = tesseract_cmd
        if isContainer:
            self.tesseract_cmd = os.environ.get("TESSERACT_PATH", "tesseract")
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        if not os.path.isfile(self.tesseract_cmd):
            raise Exception(f"Tesseract not found at {self.tesseract_cmd}")

    @cachedmethod(cache=attrgetter('cache'), key=lambda self, file_path: hashkey(file_path))
    def load_content_from_file(self, file_path: str) -> Union[str, object]:
        try:
            file_type = get_file_extension(file_path)
            
            if is_pdf_stream(file_path):
                with open(file_path, 'rb') as file:
                    return self.process_pdf(file)
            file_type = get_image_type(file_path)
            if file_type in self.SUPPORTED_FORMATS:
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
            if is_pdf_stream(stream):
                return self.process_pdf(stream)
            file_type = get_image_type(stream)
            if file_type in self.SUPPORTED_FORMATS:
                image = Image.open(stream)
                raw_text = str(pytesseract.image_to_string(image))
                self.content = raw_text
                return self.content
            else:
                raise Exception(f"Unsupported stream type: {stream}")
        except Exception as e:
            raise Exception(f"Error processing stream: {e}") from e
        
    def process_pdf(self, stream: BytesIO) -> str:
        """
        Processes a PDF by converting its pages to images and extracting text from each image.

        Args:
            stream (BytesIO): The PDF file as a BytesIO stream.

        Returns:
            str: The extracted text from all pages.
        """
        try:
            # Reset stream position
            stream.seek(0)
            file = BytesIO(stream.read())
            images = self.convert_to_images(file)
            
            # Add debug logging
            if not images:
                raise Exception("No images were extracted from PDF")
                
            extracted_text = []
            for page_number, image_bytes in images.items():
                # Convert image data to proper BytesIO stream
                if isinstance(image_bytes, bytes):
                    image_stream = BytesIO(image_bytes)
                elif isinstance(image_bytes, BytesIO):
                    image_stream = image_bytes
                else:
                    raise ValueError(f"Unexpected image data type for page {page_number}: {type(image_bytes)}")

                text = self.process_image(image_stream)
                extracted_text.append(text)

            if not extracted_text:
                raise Exception("No text was extracted from any pages")
                
            # Combine text from all pages
            self.content = "\n".join(extracted_text)
            return self.content
        except Exception as e:
            raise Exception(f"Error processing PDF: {e}") from e

    def process_image(self, image: BytesIO) -> str:
        for attempt in range(3):
            try:
                # Convert bytes to PIL Image
                pil_image = Image.open(image)
                raw_text = str(pytesseract.image_to_string(pil_image))
                if raw_text:
                    return raw_text
            except Exception as e:
                if attempt == 2:
                    raise Exception(f"Failed to process image after 3 attempts: {e}")
        return ""

    def worker(self, input_queue: Queue, output_queue: Queue):
        while True:
            image_data = input_queue.get()
            if image_data is None:  # Sentinel to indicate shutdown
                break
            try:
                # Convert bytes to BytesIO if needed
                if isinstance(image_data, bytes):
                    image_stream = BytesIO(image_data)
                elif isinstance(image_data, BytesIO):
                    image_stream = image_data
                else:
                    raise ValueError(f"Unexpected image data type: {type(image_data)}")
                    
                text = self.process_image(image_stream)
                output_queue.put((image_data, text))
            except Exception as e:
                output_queue.put((image_data, str(e)))
            input_queue.task_done()

    @cachedmethod(cache=attrgetter('cache'), key=lambda self, stream: hashkey(id(stream)))
    def load_content_from_stream_list(self, stream: BytesIO) -> List[Any]:
        images = self.convert_to_images(stream)
        input_queue = Queue()
        output_queue = Queue()

        for img in images.values():
            input_queue.put(BytesIO(img))

        threads = []
        for _ in range(4):  # Number of worker threads
            t = threading.Thread(target=self.worker, args=(input_queue, output_queue))
            t.start()
            threads.append(t)

        input_queue.join()

        for _ in range(4):
            input_queue.put(None)

        for t in threads:
            t.join()

        contents = []
        while not output_queue.empty():
            image, content = output_queue.get()
            contents.append({"image": image, "content": content})

        # put the first page at the end of the list
        contents.append(contents.pop(0))

        return contents

    @cachedmethod(cache=attrgetter('cache'), key=lambda self, input: hashkey(id(input)))
    def load_content_from_file_list(self, input: List[Union[str, BytesIO]]) -> List[Any]:
        images = self.convert_to_images(input)
        input_queue = Queue()
        output_queue = Queue()

        for img in images.values():
            input_queue.put(BytesIO(img))

        threads = []
        for _ in range(4):  # Number of worker threads
            t = threading.Thread(target=self.worker, args=(input_queue, output_queue))
            t.start()
            threads.append(t)

        input_queue.join()

        for _ in range(4):
            input_queue.put(None)

        for t in threads:
            t.join()

        contents = []
        while not output_queue.empty():
            image, content = output_queue.get()
            contents.append({"image": Image.open(image), "content": content})

        # put the first page at the end of the list
        contents.append(contents.pop(0))
        
        return contents

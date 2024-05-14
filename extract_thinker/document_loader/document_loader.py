from abc import ABC, abstractmethod
import io
from PIL import Image
from io import BytesIO
import pypdfium2 as pdfium
import concurrent.futures
from typing import Any, Dict, List, Union
from cachetools import TTLCache


class DocumentLoader(ABC):
    def __init__(self, content: Any = None, cache_ttl: int = 300):
        self.content = content
        self.file_path = None
        self.cache = TTLCache(maxsize=100, ttl=cache_ttl)

    @abstractmethod
    def load_content_from_file(self, file_path: str) -> Union[str, object]:
        pass

    @abstractmethod
    def load_content_from_stream(self, stream: BytesIO) -> Union[str, object]:
        pass

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
        page_indices = [i for i in range(len(pdf_file))]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {i: executor.submit(self.render_page, pdf_file, i, scale) for i in page_indices}

        final_images = {}
        for i, future in futures.items():
            final_images[i] = future.result()

        return final_images

    @staticmethod
    def render_page(pdf_file: pdfium.PdfDocument, page_index: int, scale: float) -> Dict[int, bytes]:
        renderer = pdf_file.render(
            pdfium.PdfBitmap.to_pil,
            page_indices=[page_index],
            scale=scale,
        )
        image_list = list(renderer)
        image = image_list[0]
        image_byte_array = BytesIO()
        image.save(image_byte_array, format="jpeg", optimize=True)
        image_byte_array = image_byte_array.getvalue()
        return {page_index: image_byte_array}

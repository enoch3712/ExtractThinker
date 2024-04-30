from abc import ABC, abstractmethod
from PIL import Image
from io import BytesIO
import pypdfium2 as pdfium
import concurrent.futures
from typing import Any, Dict, List, Union


class DocumentLoader(ABC):
    def __init__(self, content: Any = None):
        self.content = content
        self.file_path = None

    @abstractmethod
    def load_content_from_file(self, file_path: str) -> Union[str, object]:
        pass

    @abstractmethod
    def load_content_from_stream(self, stream: BytesIO) -> Union[str, object]:
        pass

    def getContent(self) -> Any:
        return self.content

    def convert_pdf_to_images(self, file_path: str, scale: float = 300 / 72) -> List[Dict[int, bytes]]:
        # Check if the file is already an image
        try:
            Image.open(file_path)
            is_image = True
        except IOError:
            is_image = False

        if is_image:
            # If it is, return it as is
            with open(file_path, "rb") as f:
                return [{0: f.read()}]

        # If it's not an image, proceed with the conversion
        pdf_file = pdfium.PdfDocument(file_path)

        page_indices = [i for i in range(len(pdf_file))]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i in page_indices:
                future = executor.submit(self.render_page, pdf_file, i, scale)
                futures.append(future)

            final_images = []
            for future in concurrent.futures.as_completed(futures):
                final_images.append(future.result())

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

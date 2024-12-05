from abc import ABC
from io import BytesIO
from typing import Any, List, Union
from PIL import Image
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader

class DocumentLoaderLLMImage(CachedDocumentLoader, ABC):
    SUPPORTED_FORMATS = ['pdf', 'jpg', 'jpeg', 'png']
    
    def __init__(self, content=None, cache_ttl=300, llm=None):
        super().__init__(content, cache_ttl)
        self.llm = llm

    def load_content_from_file(self, file_path: str) -> Union[str, object]:
        pass
        # images = self.convert_to_images(file_path)
        # results = []
        # for _, image_bytes in images.items():
        #     image_stream = BytesIO(image_bytes)
        #     results.append({"image": image_stream})
        # return results

    def load_content_from_stream(self, stream: BytesIO) -> Union[str, object]:
        pass
        # images = self.convert_to_images(stream)
        # results = []
        # for _, image_bytes in images.items():
        #     image_stream = BytesIO(image_bytes)
        #     results.append({"image": image_stream})
        # return results

    def load_content_from_stream_list(self, stream: BytesIO) -> List[Any]:
        return self.load_content_from_stream(stream)

    def load_content_from_file_list(self, file_path: str) -> List[Any]:
        return self.load_content_from_file(file_path)
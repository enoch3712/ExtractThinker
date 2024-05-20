from operator import attrgetter
import openpyxl
from typing import Union
from io import BytesIO
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader
from cachetools import cachedmethod
from cachetools.keys import hashkey


class DocumentLoaderSpreadSheet(CachedDocumentLoader):
    def __init__(self, content=None, cache_ttl=300):
        super().__init__(content, cache_ttl)

    @cachedmethod(cache=attrgetter('cache'), key=lambda self, file_path: hashkey(file_path))
    def load_content_from_file(self, file_path: str) -> Union[str, object]:
        workbook = openpyxl.load_workbook(file_path)
        sheet = workbook.active
        data = []
        for row in sheet.iter_rows(values_only=True):
            data.append(row)
        self.content = data
        return self.content

    @cachedmethod(cache=attrgetter('cache'), key=lambda self, stream: hashkey(id(stream)))
    def load_content_from_stream(self, stream: Union[BytesIO, str]) -> Union[str, object]:
        workbook = openpyxl.load_workbook(filename=BytesIO(stream.read()))
        sheet = workbook.active
        data = []
        for row in sheet.iter_rows(values_only=True):
            data.append(row)
        self.content = data
        return self.content

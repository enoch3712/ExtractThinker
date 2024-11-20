from operator import attrgetter
import openpyxl
from typing import List, Union
from io import BytesIO
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader
from cachetools import cachedmethod
from cachetools.keys import hashkey
from extract_thinker.utils import get_file_extension

class DocumentLoaderSpreadSheet(CachedDocumentLoader):
    SUPPORTED_FORMATS = ['xls', 'xlsx', 'xlsm', 'xlsb', 'odf', 'ods', 'odt', 'csv']

    def __init__(self, content=None, cache_ttl=300):
        super().__init__(content, cache_ttl)

    @cachedmethod(cache=attrgetter('cache'), key=lambda self, file_path: hashkey(file_path))
    def load_content_from_file(self, file_path: str) -> Union[str, object]:
        workbook = openpyxl.load_workbook(file_path)
        data = {}
        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            sheet_data = [self._process_row(row) for row in sheet.iter_rows(values_only=True)]
            data[sheet_name] = sheet_data
        self.content = data
        return {"data": self.content, "is_spreadsheet": True}

    @cachedmethod(cache=attrgetter('cache'), key=lambda self, stream: hashkey(id(stream)))
    def load_content_from_stream(self, stream: Union[BytesIO, str]) -> Union[str, object]:
        workbook = openpyxl.load_workbook(filename=BytesIO(stream.read()))
        data = {}
        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            sheet_data = [self._process_row(row) for row in sheet.iter_rows(values_only=True)]
            data[sheet_name] = sheet_data
        self.content = data
        return {"data": self.content, "is_spreadsheet": True}

    def _process_row(self, row):
        if all(cell in (None, '', ' ') for cell in row):
            return ["\n"]
        return [cell if cell not in (None, '', ' ') else "" for cell in row]

    def load_content_from_file_list(self, file_paths: List[str]) -> List[Union[str, object]]:
        data_list = []
        for file_path in file_paths:
            data = self.load_content_from_file(file_path)
            data_list.append(data)
        return data_list

    def load_content_from_stream_list(self, streams: List[BytesIO]) -> List[Union[str, object]]:
        data_list = []
        for stream in streams:
            data = self.load_content_from_stream(stream)
            data_list.append(data)
        return data_list

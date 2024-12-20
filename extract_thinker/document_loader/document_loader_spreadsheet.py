import openpyxl
from typing import Any, Dict, List, Union
from io import BytesIO
from operator import attrgetter
from cachetools import cachedmethod
from cachetools.keys import hashkey
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader


class DocumentLoaderSpreadSheet(CachedDocumentLoader):
    """Document loader for spreadsheet files."""
    
    SUPPORTED_FORMATS = ['xls', 'xlsx', 'xlsm', 'xlsb', 'odf', 'ods', 'odt', 'csv']

    def __init__(self, content=None, cache_ttl=300):
        super().__init__(content, cache_ttl)

    @cachedmethod(cache=attrgetter('cache'), 
                  key=lambda self, source: hashkey(source if isinstance(source, str) else source.getvalue(), self.vision_mode))
    def load(self, source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        """
        Load content from a spreadsheet and convert it to our standard format.
        Each sheet is treated as a separate "page" for consistency.

        Args:
            source: Either a file path or BytesIO stream
            
        Returns:
            List[Dict[str, Any]]: List of pages, each containing content and sheet data
        """
        if not self.can_handle(source):
            raise ValueError(f"Cannot handle source: {source}")

        try:
            # Load workbook based on source type
            if isinstance(source, str):
                workbook = openpyxl.load_workbook(source)
            else:
                # BytesIO stream
                workbook = openpyxl.load_workbook(filename=BytesIO(source.read()))

            # Convert to our standard page-based format
            pages = []
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                sheet_data = [self._process_row(row) for row in sheet.iter_rows(values_only=True)]
                
                # Create a page for each sheet
                page_dict = {
                    "content": f"Sheet: {sheet_name}",  # Sheet name as content
                    "data": sheet_data,  # Sheet data
                    "is_spreadsheet": True,  # Flag to indicate special handling
                    "sheet_name": sheet_name  # Sheet name for reference
                }
                pages.append(page_dict)

            return pages

        except Exception as e:
            raise ValueError(f"Error loading spreadsheet: {str(e)}")

    def _process_row(self, row: tuple) -> List[str]:
        """Process a row of spreadsheet data."""
        if all(cell in (None, '', ' ') for cell in row):
            return ["\n"]
        return [str(cell) if cell not in (None, '', ' ') else "" for cell in row]

    def can_handle_vision(self, source: Union[str, BytesIO]) -> bool:
        """Spreadsheet files don't support vision mode."""
        return False

    def load_content_from_file(self, file_path: str) -> Union[str, object]:
        """Legacy method for backward compatibility."""
        pages = self.load(file_path)
        data = {}
        for page in pages:
            data[page["sheet_name"]] = page["data"]
        return {"data": data, "is_spreadsheet": True}

    def load_content_from_stream(self, stream: BytesIO) -> Union[str, object]:
        """Legacy method for backward compatibility."""
        pages = self.load(stream)
        data = {}
        for page in pages:
            data[page["sheet_name"]] = page["data"]
        return {"data": data, "is_spreadsheet": True}

    def load_content_list(self, source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        """Legacy method for backward compatibility."""
        return self.load(source)

    def load_content_from_file_list(self, file_paths: List[str]) -> List[Any]:
        """Legacy method for backward compatibility."""
        all_pages = []
        for file_path in file_paths:
            pages = self.load(file_path)
            all_pages.extend(pages)
        return all_pages

    def load_content_from_stream_list(self, streams: List[BytesIO]) -> List[Any]:
        """Legacy method for backward compatibility."""
        all_pages = []
        for stream in streams:
            pages = self.load(stream)
            all_pages.extend(pages)
        return all_pages

from io import BytesIO
from operator import attrgetter
from typing import Any, List, Union
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import AnalyzeResult, DocumentPage, DocumentTable, Point
from azure.ai.formrecognizer import DocumentAnalysisClient
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader
from cachetools import cachedmethod
from cachetools.keys import hashkey


class DocumentLoaderAzureForm(CachedDocumentLoader):
    SUPPORTED_FORMATS = ["pdf", "jpeg", "jpg", "png", "bmp", "tiff", "heif", "docx", "xlsx", "pptx", "html"]
    def __init__(self, subscription_key: str, endpoint: str, content: Any = None, cache_ttl: int = 300):
        super().__init__(content, cache_ttl)
        self.subscription_key = subscription_key
        self.endpoint = endpoint
        self.credential = AzureKeyCredential(self.subscription_key)
        self.client = DocumentAnalysisClient(endpoint=self.endpoint, credential=self.credential)

    @cachedmethod(cache=attrgetter('cache'), key=lambda self, file_path: hashkey(file_path))
    def load_content_from_file(self, file_path: str) -> Union[str, object]:
        try:
            with open(file_path, "rb") as document:
                poller = self.client.begin_analyze_document("prebuilt-layout", document)
                result = poller.result()
                return self.process_result(result)
        except Exception as e:
            raise Exception(f"Error processing file: {e}") from e

    @cachedmethod(cache=attrgetter('cache'), key=lambda self, stream: hashkey(id(stream)))
    def load_content_from_stream(self, stream: Union[BytesIO, str]) -> Union[str, object]:
        try:
            poller = self.client.begin_analyze_document("prebuilt-layout", stream)
            result = poller.result()
            return self.process_result(result)
        except Exception as e:
            raise Exception(f"Error processing stream: {e}") from e

    def process_result(self, result: AnalyzeResult) -> List[dict]:
        extract_results = []
        tables = self.build_tables(result.tables)
        for page in result.pages:
            paragraphs = [p.content for p in page.lines]
            tables = self.build_tables(result.tables)
            # words_with_locations = self.process_words(page)
            # Remove lines that are present in tables
            paragraphs = self.remove_lines_present_in_tables(paragraphs, tables)
            output = {
                #"content": result.content,
                "paragraphs": paragraphs,
                #"words": words_with_locations,
                "tables": tables.get(page.page_number, [])
            }
            extract_results.append(output)
        return {"pages": extract_results}

    def remove_lines_present_in_tables(self, paragraphs: List[str], tables: dict[int, List[List[str]]]) -> List[str]:
        for table in tables.values():
            for row in table:
                for cell in row:
                    if cell in paragraphs:
                        paragraphs.remove(cell)
        return paragraphs

    def page_to_string(self, page: DocumentPage) -> str:
        page_string = ""
        for word in page.words:
            for point in word.polygon:
                page_string += f"({point.x}, {point.y}): {word.content}\n"
        return page_string

    def process_words(self, page: DocumentPage) -> List[dict]:
        words_with_locations = []

        for word in page.words:
            word_info = {
                "content": word.content,
                "bounding_box": {
                    "points": word.polygon
                },
                "page_number": page.page_number
            }
            words_with_locations.append(word_info)

        return words_with_locations

    def build_tables(self, tables: List[DocumentTable]) -> dict[int, List[List[str]]]:
        table_data = {}
        for table in tables:
            rows = []
            for row_idx in range(table.row_count):
                row = []
                for cell in table.cells:
                    if cell.row_index == row_idx:
                        row.append(cell.content)
                rows.append(row)
            # Use the page number as the key for the dictionary
            table_data[table.bounding_regions[0].page_number] = rows
        return table_data

    def build_points(self, bounding_box: List[Point]) -> List[dict]:
        return [{"x": point.x, "y": point.y} for point in bounding_box]

    def load_content_from_stream_list(self, stream: BytesIO) -> List[Any]:
        pass

    def load_content_from_file_list(self, file_path: str) -> List[Any]:
        pass

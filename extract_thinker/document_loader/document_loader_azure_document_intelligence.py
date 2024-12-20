from io import BytesIO
from operator import attrgetter
from typing import Any, Dict, List, Union
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentTable
from azure.ai.formrecognizer import DocumentAnalysisClient
from cachetools import cachedmethod
from cachetools.keys import hashkey
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader


class DocumentLoaderAzureForm(CachedDocumentLoader):
    """Loader for documents using Azure Form Recognizer."""
    
    SUPPORTED_FORMATS = ["pdf", "jpeg", "jpg", "png", "bmp", "tiff", "heif", "docx", "xlsx", "pptx", "html"]
    
    def __init__(self, subscription_key: str, endpoint: str, content: Any = None, cache_ttl: int = 300):
        super().__init__(content, cache_ttl)
        self.subscription_key = subscription_key
        self.endpoint = endpoint
        self.credential = AzureKeyCredential(self.subscription_key)
        self.client = DocumentAnalysisClient(endpoint=self.endpoint, credential=self.credential)

    @cachedmethod(cache=attrgetter('cache'),
                  key=lambda self, source: hashkey(source if isinstance(source, str) else source.getvalue(), self.vision_mode))
    def load(self, source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        """
        Load and analyze a document using Azure Form Recognizer.
        Returns a list of pages, each containing:
        - content: The text content of the page
        - tables: Any tables found on the page
        - image: The page image (if vision_mode is True)
        
        Args:
            source: Either a file path or BytesIO stream
            
        Returns:
            List[Dict[str, Any]]: List of pages with content and optional images
        """
        if not self.can_handle(source):
            raise ValueError(f"Cannot handle source: {source}")

        try:
            # Process with Azure Form Recognizer
            if isinstance(source, str):
                with open(source, "rb") as document:
                    poller = self.client.begin_analyze_document("prebuilt-layout", document)
            else:
                poller = self.client.begin_analyze_document("prebuilt-layout", source)

            result = poller.result()
            pages = []

            # Convert to our standard page-based format
            for page in result.pages:
                # Extract text content (paragraphs)
                paragraphs = [p.content for p in page.lines]
                
                # Get tables for this page
                page_tables = self.build_tables(result.tables)
                
                # Remove lines that are present in tables
                paragraphs = self.remove_lines_present_in_tables(
                    paragraphs, 
                    page_tables.get(page.page_number, [])
                )

                page_dict = {
                    "content": "\n".join(paragraphs),
                    "tables": page_tables.get(page.page_number, [])
                }

                # If vision mode is enabled, add page image
                if self.vision_mode:
                    images_dict = self.convert_to_images(source)
                    if page.page_number - 1 in images_dict:  # Azure uses 1-based page numbers
                        page_dict["image"] = images_dict[page.page_number - 1]

                pages.append(page_dict)

            return pages

        except Exception as e:
            raise ValueError(f"Error processing document: {str(e)}")

    def remove_lines_present_in_tables(self, paragraphs: List[str], tables: List[List[str]]) -> List[str]:
        """Remove any paragraph that appears in a table cell."""
        for table in tables:
            for row in table:
                for cell in row:
                    if cell in paragraphs:
                        paragraphs.remove(cell)
        return paragraphs

    def build_tables(self, tables: List[DocumentTable]) -> Dict[int, List[List[str]]]:
        """Build a dictionary of page number to tables mapping."""
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

    def can_handle_vision(self, source: Union[str, BytesIO]) -> bool:
        """Check if this loader can handle the source in vision mode."""
        return self.can_handle(source)

from io import BytesIO
from operator import attrgetter
from typing import Any, Dict, List, Union
import boto3
from cachetools import cachedmethod
from cachetools.keys import hashkey
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader
from extract_thinker.utils import is_pdf_stream


class DocumentLoaderAWSTextract(CachedDocumentLoader):
    """Loader for documents using AWS Textract."""
    
    SUPPORTED_FORMATS = ["jpeg", "png", "pdf", "tiff"]
    
    def __init__(self, aws_access_key_id=None, aws_secret_access_key=None, region_name=None, 
                 textract_client=None, content=None, cache_ttl=300):
        super().__init__(content, cache_ttl)
        if textract_client:
            self.textract_client = textract_client
        elif aws_access_key_id and aws_secret_access_key and region_name:
            self.textract_client = boto3.client(
                'textract',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name
            )
        else:
            raise ValueError("Either provide a textract_client or aws credentials (access key, secret key, and region).")

    @classmethod
    def from_client(cls, textract_client, content=None, cache_ttl=300):
        return cls(textract_client=textract_client, content=content, cache_ttl=cache_ttl)

    @cachedmethod(cache=attrgetter('cache'), 
                  key=lambda self, source: hashkey(source if isinstance(source, str) else source.getvalue(), self.vision_mode))
    def load(self, source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        """
        Load and analyze a document using AWS Textract.
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
            # Get the file bytes based on source type
            if isinstance(source, str):
                with open(source, 'rb') as file:
                    file_bytes = file.read()
            else:
                file_bytes = source.getvalue()

            # Process with Textract based on file type
            if is_pdf_stream(source) or (isinstance(source, str) and source.lower().endswith('.pdf')):
                result = self.process_pdf(file_bytes)
            else:
                result = self.process_image(file_bytes)

            # Convert to our standard page-based format
            pages = []
            for page_num, page_data in enumerate(result.get("pages", [])):
                page_dict = {
                    "content": "\n".join(page_data.get("lines", [])),
                    "tables": result.get("tables", [])  # For now, attach all tables to each page
                }

                # If vision mode is enabled, add page image
                if self.vision_mode:
                    images_dict = self.convert_to_images(source)
                    if page_num in images_dict:
                        page_dict["image"] = images_dict[page_num]

                pages.append(page_dict)

            return pages

        except Exception as e:
            raise ValueError(f"Error processing document: {str(e)}")

    def process_pdf(self, pdf_bytes: bytes) -> dict:
        """Process a PDF document with Textract."""
        for attempt in range(3):
            try:
                response = self.textract_client.analyze_document(
                    Document={'Bytes': pdf_bytes},
                    FeatureTypes=['TABLES']
                )
                return self._parse_analyze_document_response(response)
            except Exception as e:
                if attempt == 2:
                    raise ValueError(f"Failed to process PDF after 3 attempts: {e}")
        return {}

    def process_image(self, image_bytes: bytes) -> dict:
        """Process an image with Textract."""
        for attempt in range(3):
            try:
                response = self.textract_client.analyze_document(
                    Document={'Bytes': image_bytes},
                    FeatureTypes=['TABLES']
                )
                return self._parse_analyze_document_response(response)
            except Exception as e:
                if attempt == 2:
                    raise ValueError(f"Failed to process image after 3 attempts: {e}")
        return {}

    def _parse_analyze_document_response(self, response: dict) -> dict:
        """Parse Textract response into our format."""
        result = {
            "pages": [],
            "tables": []
        }
        
        current_page = {"lines": [], "words": []}
        
        for block in response['Blocks']:
            if block['BlockType'] == 'PAGE':
                if current_page["lines"]:
                    result["pages"].append(current_page)
                    current_page = {"lines": [], "words": []}
            elif block['BlockType'] == 'LINE':
                current_page["lines"].append(block['Text'])
            elif block['BlockType'] == 'WORD':
                current_page["words"].append(block['Text'])
            elif block['BlockType'] == 'TABLE':
                result["tables"].append(self._parse_table(block, response['Blocks']))
        
        if current_page["lines"]:
            result["pages"].append(current_page)
        
        return result

    def _parse_table(self, table_block: dict, blocks: List[dict]) -> List[List[str]]:
        """Parse a table from Textract response."""
        cells = [block for block in blocks if block['BlockType'] == 'CELL' 
                and block['Id'] in table_block['Relationships'][0]['Ids']]
        rows = max(cell['RowIndex'] for cell in cells)
        cols = max(cell['ColumnIndex'] for cell in cells)
        
        table = [['' for _ in range(cols)] for _ in range(rows)]
        
        for cell in cells:
            row = cell['RowIndex'] - 1
            col = cell['ColumnIndex'] - 1
            if 'Relationships' in cell:
                words = [block['Text'] for block in blocks 
                        if block['Id'] in cell['Relationships'][0]['Ids']]
                table[row][col] = ' '.join(words)
        
        return table

    def can_handle_vision(self, source: Union[str, BytesIO]) -> bool:
        """Check if this loader can handle the source in vision mode."""
        return self.can_handle(source)
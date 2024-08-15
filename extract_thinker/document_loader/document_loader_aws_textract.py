import asyncio
from io import BytesIO
from operator import attrgetter
import os
import threading
from typing import Any, List, Union
from PIL import Image
import boto3
import pdfium

from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader
from extract_thinker.utils import get_image_type

from cachetools import cachedmethod
from cachetools.keys import hashkey
from queue import Queue

SUPPORTED_IMAGE_FORMATS = ["jpeg", "png", "pdf"]

class DocumentLoaderAWSTextract(CachedDocumentLoader):
    def __init__(self, aws_access_key_id=None, aws_secret_access_key=None, region_name=None, textract_client=None, content=None, cache_ttl=300):
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

    @cachedmethod(cache=attrgetter('cache'), key=lambda self, file_path: hashkey(file_path))
    def load_content_from_file(self, file_path: str) -> Union[dict, object]:
        try:
            file_type = get_image_type(file_path)
            if file_type in SUPPORTED_IMAGE_FORMATS:
                with open(file_path, 'rb') as file:
                    file_bytes = file.read()
                if file_type == 'pdf':
                    return self.process_pdf(file_bytes)
                else:
                    return self.process_image(file_bytes)
            else:
                raise Exception(f"Unsupported file type: {file_path}")
        except Exception as e:
            raise Exception(f"Error processing file: {e}") from e

    @cachedmethod(cache=attrgetter('cache'), key=lambda self, stream: hashkey(id(stream)))
    def load_content_from_stream(self, stream: Union[BytesIO, str]) -> Union[dict, object]:
        try:
            file_type = get_image_type(stream)
            if file_type in SUPPORTED_IMAGE_FORMATS:
                file_bytes = stream.getvalue() if isinstance(stream, BytesIO) else stream
                if file_type == 'pdf':
                    return self.process_pdf(file_bytes)
                else:
                    return self.process_image(file_bytes)
            else:
                raise Exception(f"Unsupported stream type: {stream}")
        except Exception as e:
            raise Exception(f"Error processing stream: {e}") from e

    def process_image(self, image_bytes: bytes) -> dict:
        for attempt in range(3):
            try:
                response = self.textract_client.analyze_document(
                    Document={'Bytes': image_bytes},
                    FeatureTypes=['TABLES', 'FORMS', 'LAYOUT']
                )
                return self._parse_analyze_document_response(response)
            except Exception as e:
                if attempt == 2:
                    raise Exception(f"Failed to process image after 3 attempts: {e}")
        return {}

    def process_pdf(self, pdf_bytes: bytes) -> dict:
        pdf = pdfium.PdfDocument(pdf_bytes)
        result = {
            "pages": [],
            "tables": [],
            "forms": [],
            "layout": {}
        }
        for page_number in range(len(pdf)):
            page = pdf.get_page(page_number)
            pil_image = page.render().to_pil()
            img_byte_arr = BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            page_result = self.process_image(img_byte_arr)
            result["pages"].extend(page_result["pages"])
            result["tables"].extend(page_result["tables"])
            result["forms"].extend(page_result["forms"])
            for key, value in page_result["layout"].items():
                if key not in result["layout"]:
                    result["layout"][key] = []
                result["layout"][key].extend(value)
        return result

    def _parse_analyze_document_response(self, response: dict) -> dict:
        result = {
            "pages": [],
            "tables": [],
            "forms": [],
            "layout": {}
        }
        
        current_page = {"paragraphs": [], "lines": [], "words": []}
        
        for block in response['Blocks']:
            if block['BlockType'] == 'PAGE':
                if current_page["paragraphs"] or current_page["lines"] or current_page["words"]:
                    result["pages"].append(current_page)
                    current_page = {"paragraphs": [], "lines": [], "words": []}
            elif block['BlockType'] == 'LINE':
                current_page["lines"].append(block['Text'])
            elif block['BlockType'] == 'WORD':
                current_page["words"].append(block['Text'])
            elif block['BlockType'] == 'TABLE':
                result["tables"].append(self._parse_table(block, response['Blocks']))
            elif block['BlockType'] == 'KEY_VALUE_SET':
                if 'KEY' in block['EntityTypes']:
                    key = block['Text']
                    value = self._find_value_for_key(block, response['Blocks'])
                    result["forms"].append({"key": key, "value": value})
            elif block['BlockType'] in ['CELL', 'SELECTION_ELEMENT']:
                self._add_to_layout(result["layout"], block)
        
        if current_page["paragraphs"] or current_page["lines"] or current_page["words"]:
            result["pages"].append(current_page)
        
        return result

    def _parse_table(self, table_block, blocks):
        cells = [block for block in blocks if block['BlockType'] == 'CELL' and block['Id'] in table_block['Relationships'][0]['Ids']]
        rows = max(cell['RowIndex'] for cell in cells)
        cols = max(cell['ColumnIndex'] for cell in cells)
        
        table = [['' for _ in range(cols)] for _ in range(rows)]
        
        for cell in cells:
            row = cell['RowIndex'] - 1
            col = cell['ColumnIndex'] - 1
            if 'Relationships' in cell:
                words = [block['Text'] for block in blocks if block['Id'] in cell['Relationships'][0]['Ids']]
                table[row][col] = ' '.join(words)
        
        return table

    def _find_value_for_key(self, key_block, blocks):
        for relationship in key_block['Relationships']:
            if relationship['Type'] == 'VALUE':
                value_block = next(block for block in blocks if block['Id'] == relationship['Ids'][0])
                if 'Relationships' in value_block:
                    words = [block['Text'] for block in blocks if block['Id'] in value_block['Relationships'][0]['Ids']]
                    return ' '.join(words)
        return ''

    def _add_to_layout(self, layout, block):
        block_type = block['BlockType']
        if block_type not in layout:
            layout[block_type] = []
        
        layout_item = {
            'id': block['Id'],
            'text': block.get('Text', ''),
            'confidence': block['Confidence'],
            'geometry': block['Geometry']
        }
        
        if 'RowIndex' in block:
            layout_item['row_index'] = block['RowIndex']
        if 'ColumnIndex' in block:
            layout_item['column_index'] = block['ColumnIndex']
        if 'SelectionStatus' in block:
            layout_item['selection_status'] = block['SelectionStatus']
        
        layout[block_type].append(layout_item)

    def load_content_from_stream_list(self, stream: BytesIO) -> List[Any]:
        images = self.convert_to_images(stream)
        return self._process_images(images)

    def load_content_from_file_list(self, input: List[Union[str, BytesIO]]) -> List[Any]:
        images = self.convert_to_images(input)
        return self._process_images(images)

    async def _process_images(self, images: dict) -> List[Any]:
        tasks = [self.process_image(img) for img in images.values()]
        results = await asyncio.gather(*tasks)
        
        contents = []
        for (image_name, image), content in zip(images.items(), results):
            contents.append({"image": Image.open(BytesIO(image)) if isinstance(image, bytes) else image, "content": content})
        
        return contents
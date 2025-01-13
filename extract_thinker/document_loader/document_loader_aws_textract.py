from io import BytesIO
from operator import attrgetter
from typing import Any, Dict, List, Optional, Union, ClassVar
from dataclasses import dataclass, field
from cachetools import cachedmethod
from cachetools.keys import hashkey
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader
from extract_thinker.utils import is_pdf_stream


@dataclass
class TextractConfig:
    """Configuration for AWS Textract document loader.
    
    Args:
        aws_access_key_id: AWS access key ID
        aws_secret_access_key: AWS secret access key
        region_name: AWS region name
        textract_client: Pre-configured Textract client (optional)
        content: Initial content (optional)
        cache_ttl: Cache time-to-live in seconds (default: 300)
        feature_types: List of Textract feature types to analyze (default: ['TABLES'])
        max_retries: Maximum number of retries for failed requests (default: 3)
    """
    # Class level constants for allowed feature types
    ALLOWED_FEATURE_TYPES: ClassVar[List[str]] = [
        "TABLES",
        "FORMS", 
        "LAYOUT",
        "SIGNATURES"
    ]

    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    region_name: Optional[str] = None
    textract_client: Optional[Any] = None
    content: Optional[Any] = None
    cache_ttl: int = 300
    feature_types: List[str] = field(default_factory=list)  # Empty list means "Raw Text" only
    max_retries: int = 3

    def __post_init__(self):
        """Validate feature types after initialization."""
        # Validate each feature type
        invalid_features = [ft for ft in self.feature_types if ft not in self.ALLOWED_FEATURE_TYPES]
        if invalid_features:
            raise ValueError(
                f"Invalid feature type(s): {invalid_features}. "
                f"Allowed types are: {self.ALLOWED_FEATURE_TYPES} "
                f"(empty list for Raw Text only)"
            )

    @property
    def api_feature_types(self) -> List[str]:
        """Get the feature types to use in the API call."""
        # For raw text extraction, we need to use DOCUMENT
        if not self.feature_types:
            return ["DOCUMENT"]
        return self.feature_types


class DocumentLoaderAWSTextract(CachedDocumentLoader):
    """Loader for documents using AWS Textract."""
    
    SUPPORTED_FORMATS = ["jpeg", "png", "pdf", "tiff"]
    
    def __init__(self, aws_access_key_id: Union[str, TextractConfig] = None, 
                 aws_secret_access_key: Optional[str] = None, 
                 region_name: Optional[str] = None,
                 textract_client: Optional[Any] = None, 
                 content: Optional[Any] = None, 
                 cache_ttl: int = 300,
                 feature_types: Optional[List[str]] = None):
        """Initialize loader.
        
        Args:
            aws_access_key_id: Either a TextractConfig object or AWS access key ID
            aws_secret_access_key: AWS secret access key (only used if aws_access_key_id is a string)
            region_name: AWS region name (only used if aws_access_key_id is a string)
            textract_client: Pre-configured Textract client (only used if aws_access_key_id is a string)
            content: Initial content (only used if aws_access_key_id is a string)
            cache_ttl: Cache time-to-live in seconds (default: 300, only used if aws_access_key_id is a string)
            feature_types: List of Textract feature types to analyze (only used if aws_access_key_id is a string)
        """
        # Check required dependencies
        self._check_dependencies()
        
        # Handle both config-based and old-style initialization
        if isinstance(aws_access_key_id, TextractConfig):
            self.config = aws_access_key_id
        else:
            # Create config from individual parameters
            self.config = TextractConfig(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name,
                textract_client=textract_client,
                content=content,
                cache_ttl=cache_ttl,
                feature_types=feature_types or []
            )
        
        super().__init__(self.config.content, self.config.cache_ttl)
        
        if self.config.textract_client:
            self.textract_client = self.config.textract_client
        elif self.config.aws_access_key_id and self.config.aws_secret_access_key and self.config.region_name:
            boto3 = self._get_boto3()
            self.textract_client = boto3.client(
                'textract',
                aws_access_key_id=self.config.aws_access_key_id,
                aws_secret_access_key=self.config.aws_secret_access_key,
                region_name=self.config.region_name
            )
        else:
            raise ValueError("Either provide a textract_client or aws credentials (access key, secret key, and region).")

    @staticmethod
    def _check_dependencies():
        """Check if required dependencies are installed."""
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )

    def _get_boto3(self):
        """Lazy load boto3."""
        try:
            import boto3
            return boto3
        except ImportError:
            raise ImportError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )

    @classmethod
    def from_client(cls, textract_client, content=None, cache_ttl=300):
        config = TextractConfig(
            textract_client=textract_client,
            content=content,
            cache_ttl=cache_ttl
        )
        return cls(config)

    @cachedmethod(cache=attrgetter('cache'), 
                  key=lambda self, source: hashkey(source if isinstance(source, str) else source.getvalue(), self.vision_mode))
    def load(self, source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        """
        Load and analyze a document using AWS Textract.
        Returns a list of pages, each containing:
        - content: The text content of the page
        - tables: Any tables found on the page (if TABLES feature enabled)
        - forms: Any form fields found on the page (if FORMS feature enabled)
        - signatures: Any signatures found on the page (if SIGNATURES feature enabled)
        - image: The page image (if vision_mode is True)
        
        Args:
            source: Either a file path or BytesIO stream
            
        Returns:
            List[Dict[str, Any]]: List of pages with content and optional features
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
                    "tables": result.get("tables", []),  # Tables from the page
                    "forms": result.get("forms", {}).get(page_num + 1, {}),  # Forms for this page
                    "signatures": result.get("signatures", {}).get(page_num + 1, [])  # Signatures for this page
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
        for attempt in range(self.config.max_retries):
            try:
                # Use detect_document_text for raw text extraction
                if not self.config.feature_types:
                    response = self.textract_client.detect_document_text(
                        Document={'Bytes': pdf_bytes}
                    )
                else:
                    response = self.textract_client.analyze_document(
                        Document={'Bytes': pdf_bytes},
                        FeatureTypes=self.config.feature_types
                    )
                return self._parse_analyze_document_response(response)
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise ValueError(f"Failed to process PDF after {self.config.max_retries} attempts: {e}")
        return {}

    def process_image(self, image_bytes: bytes) -> dict:
        """Process an image with Textract."""
        for attempt in range(self.config.max_retries):
            try:
                # Use detect_document_text for raw text extraction
                if not self.config.feature_types:
                    response = self.textract_client.detect_document_text(
                        Document={'Bytes': image_bytes}
                    )
                else:
                    response = self.textract_client.analyze_document(
                        Document={'Bytes': image_bytes},
                        FeatureTypes=self.config.feature_types
                    )
                return self._parse_analyze_document_response(response)
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise ValueError(f"Failed to process image after {self.config.max_retries} attempts: {e}")
        return {}

    def _parse_analyze_document_response(self, response: dict) -> dict:
        """Parse Textract response into our format."""
        result = {
            "pages": [],
            "tables": [],
            "forms": {},  # Map of page number to form fields
            "signatures": {}  # Map of page number to signatures
        }
        
        current_page = {"lines": [], "words": []}
        current_page_num = 1
        
        for block in response['Blocks']:
            if block['BlockType'] == 'PAGE':
                if current_page["lines"]:
                    result["pages"].append(current_page)
                    current_page = {"lines": [], "words": []}
                current_page_num = block.get('Page', current_page_num)
            elif block['BlockType'] == 'LINE':
                current_page["lines"].append(block['Text'])
            elif block['BlockType'] == 'WORD':
                current_page["words"].append(block['Text'])
            elif block['BlockType'] == 'TABLE':
                result["tables"].append(self._parse_table(block, response['Blocks']))
            elif block['BlockType'] == 'KEY_VALUE_SET':
                if 'KEY' in block.get('EntityTypes', []):
                    # This is a form field key
                    form_data = self._parse_form_field(block, response['Blocks'])
                    if form_data:
                        page_num = block.get('Page', 1)
                        if page_num not in result["forms"]:
                            result["forms"][page_num] = {}
                        result["forms"][page_num].update(form_data)
            elif block['BlockType'] == 'SIGNATURE':
                page_num = block.get('Page', 1)
                if page_num not in result["signatures"]:
                    result["signatures"][page_num] = []
                result["signatures"][page_num].append({
                    'confidence': block.get('Confidence', 0),
                    'geometry': block.get('Geometry', {})
                })
        
        if current_page["lines"]:
            result["pages"].append(current_page)
        
        return result

    def _parse_form_field(self, key_block: dict, blocks: List[dict]) -> Dict[str, str]:
        """Parse a form field (key-value pair) from Textract response."""
        if 'Relationships' not in key_block:
            return {}

        # Find the value block for this key
        value_block = None
        key_text = ""
        
        # Get the key text
        for relationship in key_block['Relationships']:
            if relationship['Type'] == 'CHILD':
                for child_id in relationship['Ids']:
                    child_block = next((b for b in blocks if b['Id'] == child_id), None)
                    if child_block and child_block['BlockType'] == 'WORD':
                        key_text += child_block['Text'] + " "
            elif relationship['Type'] == 'VALUE':
                for value_id in relationship['Ids']:
                    value_block = next((b for b in blocks if b['Id'] == value_id), None)
                    if value_block:
                        break

        if not value_block or not key_text:
            return {}

        # Get the value text
        value_text = ""
        if 'Relationships' in value_block:
            for relationship in value_block['Relationships']:
                if relationship['Type'] == 'CHILD':
                    for child_id in relationship['Ids']:
                        child_block = next((b for b in blocks if b['Id'] == child_id), None)
                        if child_block and child_block['BlockType'] == 'WORD':
                            value_text += child_block['Text'] + " "

        return {key_text.strip(): value_text.strip()}

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
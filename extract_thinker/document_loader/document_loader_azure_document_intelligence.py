from io import BytesIO
from operator import attrgetter
from typing import Any, Dict, List, Union, Optional, ClassVar
from dataclasses import dataclass, field
from cachetools import cachedmethod
from cachetools.keys import hashkey
from extract_thinker.document_loader.cached_document_loader import CachedDocumentLoader


@dataclass
class AzureConfig:
    """Configuration for Azure Document Intelligence loader.
    
    Args:
        subscription_key: Azure subscription key
        endpoint: Azure endpoint URL
        content: Initial content (optional)
        cache_ttl: Cache time-to-live in seconds (default: 300)
        model_id: Azure model ID to use (default: "prebuilt-layout")
        max_retries: Maximum number of retries for failed requests (default: 3)
    """
    # Class level constants for allowed model IDs
    # Primary general-purpose models
    GENERAL_MODELS: ClassVar[List[str]] = [
        "prebuilt-read",      # OCR/Read: Extract printed and handwritten text
        "prebuilt-layout",    # Layout: Extract tables, selection marks, and text
        "prebuilt-document"   # General documents: Extract key-value pairs and structure
    ]
    
    # Specialized prebuilt models
    SPECIALIZED_MODELS: ClassVar[List[str]] = [
        "prebuilt-invoice",           # Invoices
        "prebuilt-receipt",           # Receipts
        "prebuilt-idDocument",        # Identity documents
        "prebuilt-businessCard",      # Business cards
        "prebuilt-tax.us.w2",         # US W2 tax forms
        "prebuilt-tax.us.1040",       # US 1040 tax forms
        "prebuilt-tax.us.1099.div",   # US 1099-DIV tax forms
        "prebuilt-tax.us.1099.int",   # US 1099-INT tax forms
        "prebuilt-tax.us.1099.misc",  # US 1099-MISC tax forms
        "prebuilt-tax.us.1099.nec",   # US 1099-NEC tax forms
        "prebuilt-tax.us.1098",       # US 1098 tax forms
        "prebuilt-tax.us.1095.c",     # US 1095-C tax forms
        "prebuilt-w8",                # W8 forms
        "prebuilt-passport",          # Passports
        "prebuilt-driverLicense",     # Driver's licenses
        "prebuilt-contract",          # Contracts
        "prebuilt-healthInsurance",   # US health insurance cards
        "prebuilt-bankStatement",     # Bank statements
        "prebuilt-payStub",           # Pay stubs
        "prebuilt-creditCard"         # Credit cards
    ]

    subscription_key: str
    endpoint: str
    content: Optional[Any] = None
    cache_ttl: int = 300
    model_id: str = "prebuilt-layout"  # Default to layout as it's most versatile
    max_retries: int = 3

    def __post_init__(self):
        """Validate model ID after initialization."""
        allowed_models = self.GENERAL_MODELS + self.SPECIALIZED_MODELS
        if self.model_id not in allowed_models:
            raise ValueError(
                f"Invalid model ID: {self.model_id}. "
                f"General purpose models: {self.GENERAL_MODELS}\n"
                f"Specialized models: {self.SPECIALIZED_MODELS}"
            )

    @property
    def is_general_model(self) -> bool:
        """Check if the current model is a general-purpose model."""
        return self.model_id in self.GENERAL_MODELS

    @property
    def is_specialized_model(self) -> bool:
        """Check if the current model is a specialized model."""
        return self.model_id in self.SPECIALIZED_MODELS


class DocumentLoaderAzureForm(CachedDocumentLoader):
    """Loader for documents using Azure Form Recognizer."""
    
    SUPPORTED_FORMATS = ["pdf", "jpeg", "jpg", "png", "bmp", "tiff", "heif", "docx", "xlsx", "pptx", "html"]
    
    def __init__(self, subscription_key: Union[str, AzureConfig], endpoint: Optional[str] = None, 
                 content: Optional[Any] = None, cache_ttl: int = 300, model_id: Optional[str] = None):
        """Initialize loader.
        
        Args:
            subscription_key: Either an AzureConfig object or the Azure subscription key
            endpoint: Azure endpoint URL (only used if subscription_key is a string)
            content: Initial content (optional, only used if subscription_key is a string)
            cache_ttl: Cache time-to-live in seconds (default: 300, only used if subscription_key is a string)
            model_id: Azure model ID to use (optional, only used if subscription_key is a string)
        """
        # Check required dependencies before any other initialization
        self._check_dependencies()
        
        # Handle both config-based and old-style initialization
        if isinstance(subscription_key, AzureConfig):
            self.config = subscription_key
        else:
            # Create config from individual parameters
            self.config = AzureConfig(
                subscription_key=subscription_key,
                endpoint=endpoint if endpoint else "",
                content=content,
                cache_ttl=cache_ttl,
                model_id=model_id if model_id else "prebuilt-layout"
            )
        
        super().__init__(self.config.content, self.config.cache_ttl)
        
        # Initialize Azure client
        self._init_azure_client()
    
    def _init_azure_client(self):
        """Initialize Azure Form Recognizer client."""
        try:
            from azure.ai.formrecognizer import DocumentAnalysisClient
            from azure.core.credentials import AzureKeyCredential
            
            self.credential = AzureKeyCredential(self.config.subscription_key)
            self.client = DocumentAnalysisClient(
                endpoint=self.config.endpoint, 
                credential=self.credential
            )
        except ImportError:
            raise ImportError(
                "Could not import azure-ai-formrecognizer python package. "
                "Please install it with `pip install azure-ai-formrecognizer`."
            )

    @staticmethod
    def _check_dependencies():
        """Check if required dependencies are installed."""
        try:
            import azure.ai.formrecognizer
            import azure.core.credentials
        except ImportError:
            raise ImportError(
                "Could not import azure-ai-formrecognizer python package. "
                "Please install it with `pip install azure-ai-formrecognizer`."
            )

    @classmethod
    def from_credentials(cls, subscription_key: str, endpoint: str, **kwargs):
        """Create loader from Azure credentials."""
        config = AzureConfig(
            subscription_key=subscription_key,
            endpoint=endpoint,
            **kwargs
        )
        return cls(config)

    @cachedmethod(cache=attrgetter('cache'),
                  key=lambda self, source: hashkey(source if isinstance(source, str) else source.getvalue(), self.vision_mode))
    def load(self, source: Union[str, BytesIO]) -> List[Dict[str, Any]]:
        """
        Load and analyze a document using Azure Form Recognizer.
        Returns a list of pages, each containing:
        - content: The text content of the page
        - tables: Any tables found on the page
        - forms: Form fields if using a forms-capable model
        - image: The page image (if vision_mode is True)
        
        Args:
            source: Either a file path or BytesIO stream
            
        Returns:
            List[Dict[str, Any]]: List of pages with content and optional features
        """
        if not self.can_handle(source):
            raise ValueError(f"Cannot handle source: {source}")

        try:
            # Process with Azure Form Recognizer
            for attempt in range(self.config.max_retries):
                try:
                    if isinstance(source, str):
                        with open(source, "rb") as document:
                            poller = self.client.begin_analyze_document(
                                self.config.model_id, 
                                document
                            )
                    else:
                        poller = self.client.begin_analyze_document(
                            self.config.model_id, 
                            source
                        )

                    result = poller.result()
                    break
                except Exception as e:
                    if attempt == self.config.max_retries - 1:
                        raise ValueError(f"Failed to process document after {self.config.max_retries} attempts: {e}")
                    continue

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
                    "tables": page_tables.get(page.page_number, []),
                    "forms": {}  # Initialize empty forms dict
                }

                # Add form fields if available (for form-capable models)
                if hasattr(result, 'key_value_pairs'):
                    page_forms = {}
                    for kv in result.key_value_pairs:
                        # Get the key and value content
                        key_content = kv.key.content if kv.key else None
                        value_content = kv.value.content if kv.value else None
                        
                        # Get the bounding regions for the key
                        key_regions = kv.key.bounding_regions if kv.key and hasattr(kv.key, 'bounding_regions') else []
                        
                        # Check if this key-value pair belongs to the current page
                        is_on_current_page = any(
                            region.page_number == page.page_number 
                            for region in key_regions
                        ) if key_regions else True  # If no regions, include in first page
                        
                        if key_content and value_content and is_on_current_page:
                            page_forms[key_content] = value_content
                    
                    page_dict["forms"] = page_forms

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

    def build_tables(self, tables: List[Any]) -> Dict[int, List[List[str]]]:
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

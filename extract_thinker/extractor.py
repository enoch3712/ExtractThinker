import asyncio
import base64
from typing import Any, Dict, List, Optional, IO, Type, Union, get_origin, get_type_hints, get_args, Annotated
from instructor.batch import BatchJob
import uuid
from pydantic import BaseModel
from extract_thinker.document_loader.document_loader_data import DocumentLoaderData
from extract_thinker.llm_engine import LLMEngine
from extract_thinker.concatenation_handler import ConcatenationHandler
from extract_thinker.document_loader.document_loader import DocumentLoader
from extract_thinker.document_loader.document_loader_llm_image import DocumentLoaderLLMImage
from extract_thinker.models.classification import Classification
from extract_thinker.models.classification_response import ClassificationResponse, ClassificationResponseInternal
from extract_thinker.llm import LLM
import os
from extract_thinker.document_loader.loader_interceptor import LoaderInterceptor
from extract_thinker.document_loader.llm_interceptor import LlmInterceptor
from concurrent.futures import ThreadPoolExecutor, as_completed
from extract_thinker.batch_job import BatchJob
from extract_thinker.models.completion_strategy import CompletionStrategy
from extract_thinker.utils import (
    add_classification_structure,
    encode_image,
    json_to_formatted_string,
    num_tokens_from_string,
)
import yaml
from copy import deepcopy
from extract_thinker.pagination_handler import PaginationHandler
from instructor.exceptions import IncompleteOutputException
from extract_thinker.exceptions import (
    ExtractThinkerError,
    InvalidVisionDocumentLoaderError,
)
from extract_thinker.utils import is_vision_error, classify_vision_error
from json import JSONDecodeError
from pydantic import ValidationError

class Extractor:
    BATCH_SUPPORTED_MODELS = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4o-2024-08-06",
        "gpt-4",
    ]

    def __init__(
        self, document_loader: Optional[DocumentLoader] = None, llm: Optional[LLM] = None
    ):
        self.document_loader: Optional[DocumentLoader] = document_loader
        self.llm: Optional[LLM] = llm
        self.file: Optional[str] = None
        self.document_loaders_by_file_type: Dict[str, DocumentLoader] = {}
        self.loader_interceptors: List[LoaderInterceptor] = []
        self.llm_interceptors: List[LlmInterceptor] = []
        self.is_classify_image: bool = False
        self._skip_loading: bool = False
        self.chunk_height: int = 1500
        self.allow_vision: bool = False

    def add_interceptor(
        self, interceptor: Union[LoaderInterceptor, LlmInterceptor]
    ) -> None:
        if isinstance(interceptor, LoaderInterceptor):
            self.loader_interceptors.append(interceptor)
        elif isinstance(interceptor, LlmInterceptor):
            self.llm_interceptors.append(interceptor)
        else:
            raise ValueError(
                "Interceptor must be an instance of LoaderInterceptor or LlmInterceptor"
            )
        
    def get_document_loader_for_file(self, source: Union[str, IO]) -> DocumentLoader:
        # If source is a string (file path), use extension-based lookup
        if isinstance(source, str):
            _, ext = os.path.splitext(source)
            loader = self.document_loaders_by_file_type.get(ext, self.document_loader)
            if loader:
                return loader
        
        # Try capability-based lookup
        if self.document_loader and self.document_loader.can_handle(source):
            return self.document_loader
        
        # Check all registered loaders
        for loader in self.document_loaders_by_file_type.values():
            if loader.can_handle(source):
                return loader
            
        raise ValueError("No suitable document loader found for the input.")

    def get_document_loader(self, source: Union[str, IO, List[Union[str, IO]]]) -> Optional[DocumentLoader]:
        """
        Retrieve the appropriate document loader for the given source.

        Args:
            source (Union[str, IO]): The input source.

        Returns:
            Optional[DocumentLoader]: The suitable document loader if available.
        """
        # First, if a primary document loader is set and it can handle the source, return it.
        if self.document_loader and self.document_loader.can_handle(source):
            return self.document_loader

        # If source is a string, attempt an extension-based lookup.
        if isinstance(source, str):
            _, ext = os.path.splitext(source)
            loader = self.document_loaders_by_file_type.get(ext)
            if loader and loader.can_handle(source):
                return loader

        # As a fallback, iterate over all registered loaders and return the first that supports the source.
        for loader in self.document_loaders_by_file_type.values():
            if loader.can_handle(source):
                return loader
        
        # if is a list, usually coming from split, return documentLoaderData
        if isinstance(source, List) or isinstance(source, dict):
            return DocumentLoaderData()
        
        # Last check, if allow vision just return the document loader llm image
        if self.allow_vision:
            return DocumentLoaderLLMImage()

        return None

    def load_document_loader(self, document_loader: DocumentLoader) -> None:
        self.document_loader = document_loader

    def load_llm(self, model: Optional[str] = None) -> None:
        if isinstance(model, LLM):
            self.llm = model
        elif model is not None:
            self.llm = LLM(model)
        else:
            raise ValueError("Either a model string or an LLM object must be provided.")

    def _validate_dependencies(self, response_model: type[BaseModel], vision: bool) -> None:
        """
        Validates that required dependencies (document_loader and llm) are present
        and that response_model is valid.

        Args:
            response_model: The Pydantic model to validate
            vision: Whether the extraction is for a vision model
        Raises:
            ValueError: If any validation fails
        """
        if self.document_loader is None and not vision:
            raise ValueError("Document loader is not set. Please set a document loader before extraction.")
            
        if self.llm is None:
            raise ValueError("LLM is not set. Please set an LLM before extraction.")
            
        if not issubclass(response_model, BaseModel) and not issubclass(response_model, Contract):
            raise ValueError("response_model must be a subclass of Pydantic's BaseModel or Contract.")

    def set_skip_loading(self, skip: bool = True) -> None:
        """Internal method to control content loading behavior"""
        self._skip_loading = skip

    def remove_images_from_content(self, content: Union[Dict[str, Any], List[Dict[str, Any]], str]) -> Union[Dict[str, Any], List[Dict[str, Any]], str]:
        """
        Remove image-related keys from the content while preserving the original structure.
        
        Args:
            content: Input content that can be a dictionary, list of dictionaries, or string
            
        Returns:
            Content with image-related keys removed, maintaining the original type
        """
        if isinstance(content, dict):
            # Create a deep copy to avoid modifying the original
            content_copy = {
                k: v for k, v in content.items()
                if k not in ('images', 'image')
            }
            return content_copy
            
        elif isinstance(content, list):
            # Handle list of dictionaries
            return [
                self.remove_images_from_content(item)
                if isinstance(item, (dict, list))
                else item
                for item in content
            ]
            
        # Return strings or other types unchanged
        return content

    def extract(
        self,
        source: Union[str, IO, List[Union[str, IO]]],
        response_model: Type[BaseModel],
        vision: bool = False,
        content: Optional[str] = None,
        completion_strategy: Optional[CompletionStrategy] = CompletionStrategy.FORBIDDEN
    ) -> Any:
        """
        Extract information from one or more sources.

        If source is a list, it loads each one, converts each to a universal format, and
        merges them as if they were a single document. This merged content is then passed
        to the extraction logic to produce a final result.

        Args:
            source: A single file path/stream or a list of them.
            response_model: A Pydantic model class for validating the extracted data.
            vision: Whether to use vision mode (affecting how content is processed).
            content: Optional extra content to prepend to the merged content.
            completion_strategy: Strategy for handling completions.

        Returns:
            The parsed result from the LLM as validated by response_model.
        """
        if isinstance(source, dict) and self.document_loader is None:
            self.document_loader = DocumentLoaderData()
        
        self._validate_dependencies(response_model, vision)
        self.extra_content = content
        self.completion_strategy = completion_strategy
        self.allow_vision = vision

        if vision:
            try:
                self._handle_vision_mode(source)
            except ValueError as e:
                raise InvalidVisionDocumentLoaderError(str(e))
        else:
            if isinstance(source, List):
                source = self.remove_images_from_content(source)

        if completion_strategy is not CompletionStrategy.FORBIDDEN:
            return self.extract_with_strategy(source, response_model, vision, completion_strategy)

        try:
            if isinstance(source, list):
                all_contents = []
                for src in source:
                    loader = self.get_document_loader(src)
                    if loader is None:
                        raise ValueError(f"No suitable document loader found for source: {src}")
                    # Load the content (e.g. text, images, metadata)
                    loaded = loader.load(src)
                    # Map to a universal format that your extraction logic understands.
                    universal = self._map_to_universal_format(loaded, vision)
                    all_contents.append(universal)
                
                # Count total pages across all documents
                total_pages = 0
                for item in all_contents:
                    # If metadata contains page count, use it
                    if isinstance(item, dict) and 'metadata' in item:
                        metadata = item.get('metadata', {})
                        if isinstance(metadata, dict) and 'num_pages' in metadata:
                            total_pages += metadata['num_pages']
                        else:
                            # Otherwise count as 1 page
                            total_pages += 1
                    else:
                        # Count as 1 page if no metadata
                        total_pages += 1
                
                # Set page count if we have an LLM
                if self.llm:
                    self.llm.set_page_count(max(1, total_pages))
                
                # Merge the text contents with a clear separator.
                merged_text = "\n\n--- Document Separator ---\n\n".join(
                    item.get("content", "") for item in all_contents
                )
                # Merge all image lists into one.
                merged_images = []
                for item in all_contents:
                    merged_images.extend(item.get("images", []))
                
                merged_content = {
                    "content": merged_text,
                    "images": merged_images,
                    "metadata": {"num_documents": len(all_contents)}
                }
                
                # Optionally, prepend any extra content provided by the caller.
                if content:
                    merged_content["content"] = content + "\n\n" + merged_content["content"]
                
                result = self._extract(merged_content, response_model, vision)
            else:
                # Single source; use existing behavior.
                if self._skip_loading:
                    # Skip loading if flag is set (content from splitting)
                    unified_content = self._map_to_universal_format(source, vision)
                else:
                    # Normal loading path
                    loader = self.get_document_loader(source)
                    if not loader:
                        raise ValueError("No suitable document loader found for the input.")
                    loaded_content = loader.load(source)
                    unified_content = self._map_to_universal_format(loaded_content, vision)
                
                # Set page count if we have an LLM
                if self.llm:
                    # Count pages from unified content
                    page_count = 1  # Default to 1 page
                    
                    # Try to get page count from metadata
                    if isinstance(unified_content, dict) and 'metadata' in unified_content:
                        metadata = unified_content.get('metadata', {})
                        if isinstance(metadata, dict) and 'num_pages' in metadata:
                            page_count = metadata['num_pages']
                    
                    self.llm.set_page_count(page_count)
                
                result = self._extract(unified_content, response_model, vision)

            return result

        except IncompleteOutputException as e:
            raise ExtractThinkerError("Incomplete output received and FORBIDDEN strategy is set") from e
        except Exception as e:
            if isinstance(e.args[0], IncompleteOutputException):
                raise ExtractThinkerError("Incomplete output received and FORBIDDEN strategy is set") from e
            
            # Handle JSON validation errors or decoding errors as incomplete output
            if isinstance(e, (ValidationError, JSONDecodeError)) or \
               "ValidationError" in str(e) or "JSONDecodeError" in str(e) or \
               "json_invalid" in str(e):
                raise ExtractThinkerError("Incomplete output received and FORBIDDEN strategy is set") from e
            
            if vision & is_vision_error(e):
                raise classify_vision_error(e, self.llm.model if self.llm else None)
            
            raise ExtractThinkerError(f"Failed to extract from source: {str(e)}")

    def _map_to_universal_format(
        self,
        content: Any,
        vision: bool = False
    ) -> Dict[str, Any]:
        """
        Maps loaded content to a universal format that _extract can process.
        The universal format is:
        {
            "content": str,      # The text content (joined from pages)
            "images": List[bytes], 
               # Optional list of image bytes if vision=True (can hold multiple)
            "metadata": {}
        }
        """
        if content is None:
            return {"content": "", "images": [], "metadata": {}}

        # If content is already in universal format, return as is
        if isinstance(content, dict) and "content" in content:
            # Ensure 'images' is a list
            if "image" in content and "images" not in content:
                # Merge single 'image' into 'images'
                content["images"] = [content["image"]] if content["image"] else []
                del content["image"]
            elif "images" in content and not isinstance(content["images"], list):
                # If 'images' is mistakenly a single byte blob, fix it
                content["images"] = [content["images"]] if content["images"] else []
            elif "images" not in content:
                content["images"] = []
            return content

        # Handle list of pages from document loader
        if isinstance(content, list):
            text_content = []
            images = []
            
            for page in content:
                if isinstance(page, dict):
                    # Extract text content
                    if 'content' in page:
                        # Add page content
                        text_content.append(page['content'])
                    
                    # Handle spreadsheet data specially
                    if page.get('is_spreadsheet', False):
                        # If this is a spreadsheet, we need to make sure we're 
                        # including the actual sheet content, not just the name
                        sheet_name = page.get('name', '') or page.get('sheet_name', '')
                        if sheet_name and not any(sheet_name in line for line in text_content):
                            text_content.append(f"Sheet: {sheet_name}")
                    
                    # Extract images if vision mode is enabled
                    if vision:
                        # If there's a list of images
                        if 'images' in page and isinstance(page['images'], list):
                            images.extend(page['images'])
                        # Or just a single 'image'
                        elif 'image' in page and page['image']:
                            images.append(page['image'])

            return {
                "content": "\n\n".join(text_content) if text_content else "",
                "images": images,
                "metadata": {"num_pages": len(content)}
            }

        # Handle string content
        if isinstance(content, str):
            return {
                "content": content,
                "images": [],
                "metadata": {}
            }

        # Handle legacy dictionary format
        if isinstance(content, dict):
            text_content = content.get("text", "")
            if isinstance(text_content, list):
                text_content = "\n".join(text_content)
            
            images = []
            if vision:
                if "images" in content and isinstance(content["images"], list):
                    images.extend(content["images"])
                elif "image" in content and content["image"]:
                    images.append(content["image"])
            
            return {
                "content": text_content,
                "images": images,
                "metadata": {k: v for k, v in content.items() 
                           if k not in ["text", "images", "image", "content"]}
            }

        raise ValueError(f"Unsupported content format: {type(content)}")

    async def extract_async(
        self,
        source: Union[str, IO, list],
        response_model: type[BaseModel],
        vision: bool = False,
        content: Optional[str] = None,
        completion_strategy: Optional[CompletionStrategy] = CompletionStrategy.FORBIDDEN
    ) -> Any:
        """
        Asynchronously extract information from the provided source.

        Args:
            source: The input source (file path, stream, or list)
            response_model: The Pydantic model to parse the response into
            vision: Whether to use vision capabilities
            content: Additional content to include in the extraction
            completion_strategy: Strategy for handling completions

        Returns:
            Parsed response matching response_model
        """
        return await asyncio.to_thread(
            self.extract,
            source,
            response_model,
            vision,
            content,
            completion_strategy
        )
    
    def extract_with_strategy(
        self, 
        source: Union[str, IO, list], 
        response_model: type[BaseModel], 
        vision: bool, 
        completion_strategy: CompletionStrategy
    ) -> Any:
        """
        Extract information using a specific completion strategy.
        
        Args:
            source: Input source (file path, stream, or list)
            response_model: Pydantic model for response parsing
            vision: Whether to use vision capabilities
            completion_strategy: Strategy for handling completions
            
        Returns:
            Parsed response matching response_model
        """
        # If source is already a list, use it directly
        if isinstance(source, list):
            content = source
        else:
            # Get appropriate document loader
            document_loader = self.get_document_loader(source)
            if document_loader is None:
                raise ValueError("No suitable document loader found for the input.")

            # Load content using list method
            content = document_loader.load(source)

        # Handle based on strategy
        if completion_strategy == CompletionStrategy.PAGINATE:
            handler = PaginationHandler(self.llm)
            return handler.handle(content, response_model, vision, self.extra_content)
        elif completion_strategy == CompletionStrategy.CONCATENATE:
            # For concatenate strategy, we still use PaginationHandler but merge results
            handler = ConcatenationHandler(self.llm)
            return handler.handle(content, response_model, vision, self.extra_content)
        else:
            raise ValueError(f"Unsupported completion strategy: {completion_strategy}")

    def _build_classification_message_content(self, classifications: List[Classification]) -> List[Dict[str, str]]:
        """
        Build message content for all classifications with their images.
        
        Args:
            classifications: List of Classification objects containing name, description and image
            
        Returns:
            List of content items (text and images) for the message
        """
        message_content = []
        for classification in classifications:
            if not classification.image:
                raise ValueError(f"Image required for classification '{classification.name}' but not found.")
            
            message_content.extend([
                {
                    "type": "text",
                    "text": f"{classification.name}: {classification.description}",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + encode_image(classification.image)
                    },
                },
            ])
        
        return message_content

    def _classify(
        self,
        content: Any,
        classifications: List[Classification]
    ) -> ClassificationResponse:
        """
        Internal method to perform classification using LLM.
        
        Args:
            content: The content to classify
            classifications: List of Classification objects
            
        Returns:
            ClassificationResponse object with the chosen classification name and confidence
        """
        # If there's no vision or no images, keep the existing single-prompt approach
        if not self.is_classify_image:
            return self._classify_text_only(content, classifications)

        # Otherwise, we do an "ask one-by-one" approach for images.
        best_classification = None
        best_confidence = 0

        # Validate and extract image from content
        if isinstance(content, list):
            # Handle list of pages
            images = []
            for item in content:
                if isinstance(item, dict) and 'image' in item:
                    images.append(item['image'])
            if not images:
                raise ValueError("No images found in content for vision-based classification.")
            # For now, just use the first image
            image_data = images[0]
        elif isinstance(content, dict) and 'image' in content:
            # Handle single page/document
            image_data = content['image']
        else:
            raise ValueError("No valid image data found in content for vision-based classification.")
        
        # Convert image data to base64
        doc_image_b64 = base64.b64encode(image_data).decode('utf-8') if isinstance(image_data, bytes) else image_data

        for classification in classifications:
            # If classification has no reference image, or user wants no comparison:
            if not classification.image:
                # We can skip or treat it as not matched
                # Or we can do some minimal prompt that just says "Does doc match classification X?"
                # Here, let's do a minimal prompt with doc image alone.
                partial_result = self._classify_one_image_no_ref(doc_image_b64, classification)
            else:
                # Compare doc image vs classification reference image
                partial_result = self._classify_one_image_with_ref(
                    doc_image_b64, 
                    classification.image, 
                    classification
                )

            # If partial_result is a ClassificationResponse, check confidence
            if partial_result and partial_result.confidence is not None:
                if partial_result.confidence > best_confidence:
                    best_confidence = partial_result.confidence
                    best_classification = partial_result

        if best_classification is None:
            # fallback
            best_classification = ClassificationResponse(
                name="Unknown",
                confidence=1
            )

        return best_classification

    def _classify_one_image_with_ref(
        self, 
        doc_image_b64: str, 
        ref_image: Union[str, bytes], 
        classification: Classification
    ) -> ClassificationResponse:
        """
        Classify doc_image_b64 as either matching or not matching
        the classification's reference image. Return name/confidence.
        """
        # Convert classification.reference image to base64 if needed:
        if isinstance(ref_image, str) and os.path.isfile(ref_image):
            ref_image_b64 = encode_image(ref_image)
        elif isinstance(ref_image, bytes):
            ref_image_b64 = base64.b64encode(ref_image).decode("utf-8")
        else:
            ref_image_b64 = encode_image(ref_image)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a server API that compares two images and decides if the doc image matches "
                    "this classification. Return a valid JSON"
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"## Classification to check: {classification.name}\n"
                            f"Description: {classification.description}\n"
                            "## Reference Image (the classification example)"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{ref_image_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "## Document Image to classify",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{doc_image_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Return JSON => {\"name\": \"<classification>\", \"confidence\": <1..10>}"
                    }
                ]
            }
        ]

        result = self.llm.request(messages, ClassificationResponseInternal)
        return ClassificationResponse(
            name=classification.name,
            confidence=result.confidence,
            classification=classification
        )

    def _classify_one_image_no_ref(
        self,
        doc_image_b64: str,
        classification: Classification
    ) -> ClassificationResponse:
        """
        Minimal fallback if user didn't provide a reference image
        but we still want a numeric confidence if doc_image matches the classification.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a server API that sees if this single image matches classification. "
                    "Output JSON => {\"name\": <classification>, \"confidence\": <1..10>} "
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"## Classification to check: {classification.name}\n"
                            f"Description: {classification.description}\n"
                            "Return JSON => {\"name\": <classification>, \"confidence\": <1..10>}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{doc_image_b64}"
                        }
                    }
                ]
            }
        ]

        result = self.llm.request(messages, ClassificationResponseInternal)
        return ClassificationResponse(
            name=classification.name,
            confidence=result.confidence,
            classification=classification
        )

    def _classify_text_only(
        self,
        content: Any,
        classifications: List[Classification]
    ) -> ClassificationResponse:
        """
        Original approach for text-based classification:
          a single prompt that enumerates all possible classes and tries to parse JSON.
        """
        # Build classification info with structure
        classification_info = "\n".join(
            f"{c.name}: {c.description} \n{add_classification_structure(c)}"
            for c in classifications
        )

        messages = [
            {
                "role": "system",
                "content": "You are a server API that receives document information "
                "and returns specific fields in JSON format.\n",
            },
            {
                "role": "user",
                "content": (
                    f"##Content\n{content}\n##Classifications\n"
                    f"#if contract present, each field present increase confidence level\n"
                    f"{classification_info}\n"
                    "#Don't use contract structure, just to help on the ClassificationResponse\n"
                    "Output Example: \n"
                    "{\r\n\t\"name\": \"DMV Form\",\r\n\t\"confidence\": 8\r\n}"
                    "\n\n##ClassificationResponse JSON Output\n"
                )
            }
        ]

        # Get the internal response first
        response = self.llm.request(messages, ClassificationResponseInternal)
        
        # Find and set the matching classification object
        matched_classification = None
        for classification in classifications:
            # Make sure we're doing an exact string match
            if classification.name.strip().lower() == response.name.strip().lower():
                matched_classification = classification
                break
                
        return ClassificationResponse(
            name=matched_classification.name,
            confidence=response.confidence,
            classification=matched_classification
        )

    def classify(
        self,
        input: Union[str, IO],
        classifications: List[Classification],
        vision: bool = False,
    ) -> ClassificationResponse:
        """
        Classify the input using the provided classifications.
        
        Args:
            input: The input to classify (file path, stream, or image data)
            classifications: List of Classification objects
            vision: Whether to use vision capabilities for classification
            
        Returns:
            ClassificationResponse object
        """
        # Get appropriate document loader and configure it
        document_loader = self.get_document_loader_for_file(input)
        if document_loader is None:
            raise ValueError("No suitable document loader found for the input.")
            
        self.is_classify_image = vision
        if vision:
            document_loader.set_vision_mode(True)
            
        # Load and process the content
        content = document_loader.load(input)
        
        # For vision mode, ensure we have image data
        if vision and isinstance(content, dict) and 'image' not in content:
            content = {'image': encode_image(input)}
            
        return self._classify(content, classifications)

    async def classify_async(
        self, input: Union[str, IO], classifications: List[Classification], vision: bool = False
    ) -> ClassificationResponse:
        """
        Asynchronously classify the input using the provided classifications.
        
        Args:
            input: The input to classify (file path, stream, or image data)
            classifications: List of Classification objects
            vision: Whether to use vision capabilities for classification
            
        Returns:
            ClassificationResponse object
        """
        return await asyncio.to_thread(self.classify, input, classifications, vision)

    def _extract_with_splitting(
            self,
            content,
            file_or_stream,
            response_model,
            vision,
            is_stream,
            max_tokens_per_request,
            base_messages,
        ):
            chunks = self.split_content(content, max_tokens_per_request)
            results = []
            with ThreadPoolExecutor() as executor:
                # Prepare the messages for each chunk
                futures = [
                    executor.submit(
                        self.llm.request,
                        deepcopy(base_messages) + [
                            {
                                "role": "user",
                                "content": "##Content\n\n" + chunk,
                            }
                        ],
                        response_model,
                    )
                    for chunk in chunks
                ]

                # Collect the results as they complete
                for future in as_completed(futures):
                    try:
                        response = future.result()
                        results.append(response)
                    except Exception as e:
                        # Handle exceptions as needed
                        # For example, log the error and continue
                        print(f"Error processing chunk: {e}")
                        # Optionally, append a default value or skip
                        results.append(None)

            return self.aggregate_results(results, response_model)

    def split_content(
        self, content: Union[str, Dict[Any, Any]], max_tokens: int
    ) -> List[str]:
        if isinstance(content, dict):
            content_str = yaml.dump(content)
        else:
            content_str = content

        paragraphs = content_str.split("\n\n")
        chunks = []
        current_chunk = ""
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            tokens_in_paragraph = num_tokens_from_string(paragraph)
            tokens_in_current_chunk = num_tokens_from_string(current_chunk)
            if tokens_in_current_chunk + tokens_in_paragraph + 1 > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    # Paragraph is too long, need to split it further
                    chunks.append(paragraph)
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def aggregate_results(self, results: List[Any], response_model: type[BaseModel]) -> Any:
        if len(results) == 1:
            return results[0]

        # Initialize an empty dict to accumulate the results
        aggregated_dict: Dict[str, Any] = {}
        for result in results:
            if isinstance(result, BaseModel):
                result_dict = result.model_dump()
            elif isinstance(result, dict):
                result_dict = result
            else:
                raise ValueError(f"Unsupported result type: {type(result)}")

            for key, value in result_dict.items():
                if key not in aggregated_dict:
                    aggregated_dict[key] = value
                else:
                    # Retrieve the field type from the response_model
                    field_info = response_model.model_fields.get(key)
                    if field_info is not None:
                        field_type = field_info.annotation
                    else:
                        # Default to the type of the existing value if field is not defined
                        field_type = type(aggregated_dict[key])

                    # Check if the field is a list type
                    if get_origin(field_type) is list or isinstance(aggregated_dict[key], list):
                        if not isinstance(aggregated_dict[key], list):
                            aggregated_dict[key] = [aggregated_dict[key]]
                        if isinstance(value, list):
                            aggregated_dict[key].extend(value)
                        else:
                            aggregated_dict[key].append(value)
                    else:
                        # For scalar fields, keep the existing value
                        # Optionally, check if the new value is the same as the existing one
                        if aggregated_dict[key] != value:
                            # Decide how to handle conflicting scalar values
                            # For this example, we'll keep the first value and ignore the rest
                            pass  # Do nothing or log a warning if needed

        # Create an instance of the response_model with the aggregated_dict
        return response_model(**aggregated_dict)

    def extract_batch(
        self,
        source: Union[str, IO, List[Union[str, IO]]],
        response_model: Type[BaseModel],
        vision: bool = False,
        content: Optional[str] = None,
        output_file_path: Optional[str] = None,
        batch_file_path: Optional[str] = None,
    ) -> BatchJob:
        """
        Extracts information from a source or list of sources using batch processing.

        Args:
            source: A single source (file path or IO stream) or a list of sources.
            response_model: The Pydantic model to parse the response into.
            vision: Whether to use vision capabilities (processing images).
            content: Additional content to include in the extraction.

        Returns:
            A BatchJob object to monitor and retrieve batch processing results.

        Raises:
            ValueError: If batch processing is not supported by the current LLM configuration
        """
        if not self.llm:
            raise ValueError("LLM is not set. Please set an LLM before extraction.")

        # Check if using pydantic-ai backend
        if self.llm.backend == LLMEngine.PYDANTIC_AI:
            raise ValueError(
                "Batch processing is not supported with the PYDANTIC_AI backend. "
                "Please use GPT4o models and default backend for batch operations."
            )

        if not self.can_handle_batch():
            raise ValueError(
                f"Model {self.llm.model} does not support batch processing. "
                f"Supported models: {', '.join(self.BATCH_SUPPORTED_MODELS)}"
            )

        # Create batch directory if it doesn't exist
        batch_dir = os.path.join(os.getcwd(), "extract_thinker_batch")
        os.makedirs(batch_dir, exist_ok=True)

        # Generate unique paths if not provided
        unique_id = str(uuid.uuid4())
        if output_file_path is None:
            new_output_file_path = os.path.join(batch_dir, f"output_{unique_id}.jsonl")
        else:
            new_output_file_path = output_file_path
            
        if batch_file_path is None:
            new_batch_file_path = os.path.join(batch_dir, f"input_{unique_id}.jsonl")
        else:
            new_batch_file_path = batch_file_path
        
        # Check if provided paths exist
        for path in [new_output_file_path, new_batch_file_path]:
            if os.path.exists(path):
                raise ValueError(f"File already exists: {path}")

        self.extra_content = content
        
        # Ensure that sources is a list
        if not isinstance(source, list):
            sources = [source]
        else:
            sources = source

        # Set page count for batch processing if we have an LLM
        if self.llm:
            # Estimate pages based on the number of sources
            if isinstance(source, list):
                estimated_pages = len(source)
            else:
                estimated_pages = 1
                
            self.llm.set_page_count(estimated_pages)
        
        def get_messages():
            for idx, src in enumerate(sources):
                messages = [
                    {
                        "role": "system",
                        "content": "You are a server API that receives document information and returns specific fields in JSON format.",
                    }
                ]

                if vision:
                    # Handle vision content
                    if isinstance(src, str):
                        if os.path.exists(src):
                            with open(src, "rb") as f:
                                image_data = f.read()
                        else:
                            raise ValueError(f"File {src} does not exist.")
                    elif isinstance(src, IO):
                        image_data = src.read()
                    else:
                        raise ValueError("Invalid source type for vision data.")

                    encoded_image = base64.b64encode(image_data).decode("utf-8")
                    image_content = f"data:image/jpeg;base64,{encoded_image}"
                    
                    message_content = [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_content
                            }
                        }
                    ]
                    if self.extra_content:
                        message_content.insert(0, {"type": "text", "text": self.extra_content})
                    
                    messages.append({
                        "role": "user",
                        "content": message_content
                    })
                else:
                    # Handle regular content
                    if isinstance(src, str):
                        if os.path.exists(src):
                            pages = self.document_loader.load(src)
                            content_data = self._format_pages_to_content(pages)
                        else:
                            content_data = src  # Assume src is the text content
                    elif isinstance(src, IO):
                        pages = self.document_loader.load(src)
                        content_data = self._format_pages_to_content(pages)
                    else:
                        raise ValueError("Invalid source type.")

                    message_text = f"##Content\n\n{content_data}"
                    if self.extra_content:
                        message_text = f"##Extra Content\n\n{self.extra_content}\n\n" + message_text

                    messages.append({
                        "role": "user",
                        "content": message_text
                    })
            
                yield messages  # Yield the complete messages list
        
        # Create batch job with the message generator
        batch_job = BatchJob(
            messages_batch=get_messages(),
            model=self.llm.model,
            response_model=response_model,
            file_path=new_batch_file_path,
            output_path=new_output_file_path
        )

        return batch_job

    def can_handle_batch(self) -> bool:
        """
        Checks if the current LLM model supports batch processing.
        
        Returns:
            bool: True if batch processing is supported, False otherwise.
        """
        if not self.llm or not self.llm.model:
            return False
            
        return any(
            supported_model in self.llm.model.lower() 
            for supported_model in self.BATCH_SUPPORTED_MODELS
        )

    def _extract(
        self,
        content: Optional[Union[Dict[str, Any], List[Any], str]],
        response_model: Any,
        vision: bool = False,
    ) -> Any:
        """Extract information from the content using the LLM."""
        # Call all the LLM interceptors before calling the LLM
        for interceptor in self.llm_interceptors:
            interceptor.intercept(self.llm)

        # Build messages
        messages = self._build_messages(self._build_message_content(content, vision))

        if self.extra_content is not None:
            self._add_extra_content(messages)

        # Handle based on completion strategy
        try:
            if self.completion_strategy == CompletionStrategy.PAGINATE:
                handler = PaginationHandler(self.llm)
                return handler.handle(messages, response_model, vision, self.extra_content)
            elif self.completion_strategy == CompletionStrategy.CONCATENATE:
                handler = ConcatenationHandler(self.llm)
                return handler.handle(messages, response_model, vision, self.extra_content)
            elif self.completion_strategy == CompletionStrategy.FORBIDDEN:
                return self.llm.request(messages, response_model)
            else:
                raise ValueError(f"Unsupported completion strategy: {self.completion_strategy}")
        except IncompleteOutputException as e:
            if self.completion_strategy == CompletionStrategy.FORBIDDEN:
                raise ExtractThinkerError("Incomplete output received and FORBIDDEN strategy is set") from e
            raise e

    def _build_message_content(
        self,
        content: Optional[Union[Dict[str, Any], List[Any], str]],
        vision: bool,
    ) -> Union[List[Dict[str, Any]], List[str]]:
        """
        Build the message content based on the content and vision flag.

        Args:
            content: The content to process.
            vision: Whether to process vision content.

        Returns:
            A list representing the message content.
        """
        message_content: Union[List[Dict[str, Any]], List[str]] = []

        if content is None:
            return message_content

        if vision:
            content_data = self._process_content_data(content)
            if content_data:
                message_content.append({
                    "type": "text",
                    "text": "##Content\n\n" + content_data
                })
            self._add_images_to_message_content(content, message_content)
        else:
            content_str = self._convert_content_to_string(content)
            if content_str:
                message_content.append("##Content\n\n" + content_str)

        return message_content

    def _process_content_data(
        self,
        content: Union[Dict[str, Any], List[Any], str],
    ) -> Optional[str]:
        """
        Process content data by filtering out images and converting to a string.
        Handles both legacy format and new page-based format from document loaders.

        Args:
            content: The content to process.

        Returns:
            A string representation of the content.
        """
        if isinstance(content, list):
            # Handle new page-based format from document loaders
            # Concatenate all page contents
            page_texts = []
            for page in content:
                if isinstance(page, dict):
                    page_text = page.get('content', '')
                    if page_text:
                        page_texts.append(page_text)
            return "\n\n".join(page_texts) if page_texts else None
            
        elif isinstance(content, dict):
            # Handle legacy dictionary format
            filtered_content = {
                k: v for k, v in content.items()
                if k != 'images' and k != 'image' and not hasattr(v, 'read')
            }
            if filtered_content.get("is_spreadsheet", False):
                content_str = json_to_formatted_string(filtered_content.get("data", {}))
            else:
                content_str = yaml.dump(filtered_content, default_flow_style=True)
            return content_str
        elif isinstance(content, str):
            return content
        return None

    def _convert_content_to_string(
        self,
        content: Union[Dict[str, Any], List[Any], str]
    ) -> Optional[str]:
        """
        Convert content to a string representation.
        Handles both legacy format and new page-based format from document loaders.

        Args:
            content: The content to convert.

        Returns:
            A string representation of the content.
        """
        if isinstance(content, list):
            # Handle new page-based format
            page_texts = []
            for page in content:
                if isinstance(page, dict):
                    page_text = page.get('content', '')
                    if page_text:
                        page_texts.append(page_text)
            return "\n\n".join(page_texts) if page_texts else None
        elif isinstance(content, dict):
            if content.get("is_spreadsheet", False):
                return json_to_formatted_string(content.get("data", {}))
            else:
                return yaml.dump(content, default_flow_style=True)
        elif isinstance(content, str):
            return content
        return None

    def _add_images_to_message_content(
        self,
        content: Union[Dict[str, Any], List[Any]],
        message_content: List[Dict[str, Any]],
    ) -> None:
        if isinstance(content, list):
            for page in content:
                if isinstance(page, dict):
                    if 'image' in page:
                        self._append_images(page['image'], message_content)
                    if 'images' in page:
                        self._append_images(page['images'], message_content)
        elif isinstance(content, dict):
            if 'image' in content:
                self._append_images(content['image'], message_content)
            if 'images' in content:
                self._append_images(content['images'], message_content)

    def _append_images(
        self,
        image_data: Union[Dict[str, Any], List[Any], Any],
        message_content: List[Dict[str, Any]],
    ) -> None:
        """
        Append images to the message content.

        Args:
            image_data: The image data to process. Can be:
                - A dictionary with 'image' or 'images' keys
                - A list of images
                - A single image
            message_content: The message content to append images to.
        """
        if not image_data:
            return

        images_list = []
        if isinstance(image_data, dict):
            # Handle dictionary format
            if "images" in image_data:
                # If "images" key exists, it should be a list of images
                if isinstance(image_data["images"], list):
                    images_list.extend(image_data["images"])
                else:
                    # Single image in "images" key
                    images_list.append(image_data["images"])
            elif "image" in image_data and image_data["image"] is not None:
                # Single image in "image" key
                images_list.append(image_data["image"])
        elif isinstance(image_data, list):
            # Process list of images or image dictionaries
            for item in image_data:
                if isinstance(item, dict):
                    # Handle nested image dictionaries
                    if "images" in item and isinstance(item["images"], list):
                        images_list.extend(item["images"])
                    elif "image" in item and item["image"] is not None:
                        images_list.append(item["image"])
                else:
                    # Raw image data
                    images_list.append(item)
        else:
            # Single raw image
            images_list.append(image_data)

        # Process all collected images
        for img in images_list:
            if img is not None:  # Skip None values
                base64_image = encode_image(img)
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })

    def _build_messages(
        self,
        message_content: Union[List[Dict[str, Any]], List[str]],
    ) -> List[Dict[str, Any]]:
        """
        Build the messages to send to the LLM.

        Args:
            message_content: The content of the message.

        Returns:
            A list of messages.
        """
        system_message = {
            "role": "system",
            "content": "You are a server API that receives document information and returns specific fields in JSON format.",
        }

        if self.allow_vision:
            messages = [
                system_message,
                {
                    "role": "user",
                    "content": message_content
                }
            ]
        else:
            messages = [system_message]
            if message_content:
                messages.append({
                    "role": "user",
                    "content": "".join(message_content)
                })

        return messages

    def _add_extra_content(
        self,
        messages: List[Dict[str, Any]],
    ) -> None:
        """
        Add extra content to the messages.

        Args:
            messages: The list of messages to modify.
        """
        if isinstance(self.extra_content, dict):
            extra_content_str = yaml.dump(self.extra_content)
        else:
            extra_content_str = self.extra_content

        extra_message = {
            "role": "user",
            "content": "##Extra Content\n\n" + extra_content_str
        }

        # Insert the extra content after the system message
        messages.insert(1, extra_message)

    def loadfile(self, file):
        self.file = file
        return self

    def loadstream(self, stream):
        return self

    def _handle_vision_mode(self, source: Union[str, IO, list]) -> None:
        """
        Sets up the document loader or raises error if LLM or loader doesn't support vision.
        If no document loader is available but vision is needed, falls back to DocumentLoaderLLMImage.
        """
        if self.document_loader:
            self.document_loader.set_vision_mode(True)
            return

        # No document loader available, create a new DocumentLoaderLLMImage
        self.document_loader = DocumentLoaderLLMImage(llm=self.llm)
        self.document_loader.set_vision_mode(True)

    def _format_pages_to_content(self, pages: List[Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
        """
        Convert pages to content format.
        Returns either a string for regular content or a dict for spreadsheet data.
        """
        if not pages:
            return ""
            
        # Handle spreadsheet data specially
        if any(page.get("is_spreadsheet") for page in pages):
            data = {}
            for page in pages:
                if "sheet_name" in page and "data" in page:
                    data[page["sheet_name"]] = page["data"]
            return {"data": data, "is_spreadsheet": True}
            
        # For regular content, join all page contents
        return "\n\n".join(page.get("content", "") for page in pages)

    def enable_thinking_mode(self, enable: bool = True) -> 'Extractor':
        """Enable thinking mode for the LLM if supported.
        
        Args:
            enable (bool): Whether to enable thinking mode
            
        Returns:
            Extractor: self for method chaining
        """
        if self.llm is None:
            raise ValueError("LLM must be set before enabling thinking mode")
            
        self.enable_thinking = enable
        self.llm.set_thinking(enable)
        return self
    
    def set_page_count(self, page_count: int) -> 'Extractor':
        """Set the page count for token calculation when thinking is enabled.
        
        Args:
            page_count (int): Number of pages in the document
            
        Returns:
            Extractor: self for method chaining
        """
        if self.llm is None:
            raise ValueError("LLM must be set before setting page count")
            
        self.llm.set_page_count(page_count)
        return self
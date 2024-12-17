import asyncio
import base64
from typing import Any, Dict, List, Optional, IO, Type, Union, get_origin
from instructor.batch import BatchJob
import uuid
import litellm
from pydantic import BaseModel
from extract_thinker.concatenation_handler import ConcatenationHandler
from extract_thinker.document_loader.document_loader import DocumentLoader
from extract_thinker.document_loader.document_loader_llm_image import DocumentLoaderLLMImage
from extract_thinker.models.classification import Classification
from extract_thinker.models.classification_response import ClassificationResponse
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

    def get_document_loader(self, source: Union[str, IO]) -> Optional[DocumentLoader]:
        """
        Retrieve the appropriate document loader for the given source.

        Args:
            source (Union[str, IO]): The input source.

        Returns:
            Optional[DocumentLoader]: The suitable document loader if available.
        """
        if isinstance(source, str):
            _, ext = os.path.splitext(source)
            return self.document_loaders_by_file_type.get(ext, self.document_loader)
        elif hasattr(source, 'read'):
            # Implement logic to determine the loader based on the stream if necessary
            return self.document_loader
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

    def extract(
        self,
        source: Union[str, IO, list],
        response_model: type[BaseModel],
        vision: bool = False,
        content: Optional[str] = None,
        completion_strategy: Optional[CompletionStrategy] = CompletionStrategy.FORBIDDEN
    ) -> Any:
        """
        Extract information from the provided source.
        """
        self._validate_dependencies(response_model, vision)
        self.extra_content = content
        self.completion_strategy = completion_strategy

        if vision:
            self._handle_vision_mode(source)

        if completion_strategy is not CompletionStrategy.FORBIDDEN:
            return self.extract_with_strategy(source, response_model, vision, completion_strategy)

        try:
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

            return self._extract(unified_content, response_model, vision)
        except IncompleteOutputException as e:
            raise ValueError("Incomplete output received and FORBIDDEN strategy is set") from e
        except Exception as e:
            if isinstance(e.args[0], IncompleteOutputException):
                raise ValueError("Incomplete output received and FORBIDDEN strategy is set") from e
            raise ValueError(f"Failed to extract from source: {str(e)}")

    def _map_to_universal_format(
        self,
        content: Any,
        vision: bool = False
    ) -> Dict[str, Any]:
        """
        Maps loaded content to a universal format that _extract can process.
        The universal format is:
        {
            "content": str,  # The text content
            "images": List[bytes],  # Optional list of image bytes if vision=True
            "metadata": Dict[str, Any]  # Optional metadata
        }
        """
        if content is None:
            return {"content": "", "images": [], "metadata": {}}

        # If content is already in universal format, return as is
        if isinstance(content, dict) and "content" in content:
            return content

        # Handle list of pages from document loader
        if isinstance(content, list):
            text_content = []
            images = []
            
            for page in content:
                if isinstance(page, dict):
                    # Extract text content
                    if 'content' in page:
                        text_content.append(page['content'])
                    # Extract images if vision mode is enabled
                    if vision and 'image' in page:
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
            
            return {
                "content": text_content,
                "images": content.get("images", []) if vision else [],
                "metadata": {k: v for k, v in content.items() 
                           if k not in ["text", "images", "content"]}
            }

        raise ValueError(f"Unsupported content format: {type(content)}")

    async def extract_async(
        self,
        source: Union[str, IO, list],
        response_model: type[BaseModel],
        vision: bool = False,
    ) -> Any:
        return await asyncio.to_thread(self.extract, source, response_model, vision)
    
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

    def _classify(
        self, content: Any, classifications: List[Classification]
    ):
        """
        Internal method to perform classification using LLM.
        
        Args:
            content: The content to classify
            classifications: List of Classification objects
            
        Returns:
            ClassificationResponse object
        """
        messages = [
            {
                "role": "system",
                "content": "You are a server API that receives document information "
                "and returns specific fields in JSON format.\n",
            },
        ]

        # Common classification structure for both image and non-image cases
        classification_info = "\n".join(
            f"{c.name}: {c.description} \n{add_classification_structure(c)}"
            for c in classifications
        )

        if self.is_classify_image:
            input_data = (
                f"##Take the last image, and compare to the several images provided. Then classify according to the classification attached to the image\n"
                f"##Classifications\n{classification_info}\n"
                + "Output Example: \n"
                + "{\r\n\t\"name\": \"DMV Form\",\r\n\t\"confidence\": 8\r\n}"
                + "\n\n##ClassificationResponse JSON Output\n"
            )
            
            # Add input data as first message
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": input_data,
                    },
                ],
            })

            # Add classification images if present
            for classification in classifications:
                if classification.image:
                    if not litellm.supports_vision(model=self.llm.model):
                        raise ValueError(
                            f"Model {self.llm.model} is not supported for vision, since it's not a vision model."
                        )

                    messages.append({
                        "role": "user",
                        "content": [
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
                        ],
                    })
                else:
                    raise ValueError(
                        f"Image required for classification '{classification.name}' but not found."
                    )

            # Add the content image to be classified
            if isinstance(content, dict) and 'image' in content:
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "##classify",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64," + content['image']
                            },
                        },
                    ],
                })
        else:
            input_data = (
                f"##Content\n{content}\n##Classifications\n#if contract present, each field present increase confidence level\n"
                f"{classification_info}\n"
                + "#Don't use contract structure, just to help on the ClassificationResponse\nOutput Example: \n"
                + "{\r\n\t\"name\": \"DMV Form\",\r\n\t\"confidence\": 8\r\n}"
                + "\n\n##ClassificationResponse JSON Output\n"
            )
            messages.append({"role": "user", "content": input_data})

        return self.llm.request(messages, ClassificationResponse)

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
        """
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

        if vision and not litellm.supports_vision(model=self.llm.model):
            raise ValueError(
                f"Model {self.llm.model} is not supported for vision."
            )

        # Build messages
        messages = self._build_messages(self._build_message_content(content, vision), vision)

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
                raise ValueError("Incomplete output received and FORBIDDEN strategy is set") from e
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
        """
        Add images to the message content.
        Handles both legacy format and new page-based format from document loaders.

        Args:
            content: The content containing images.
            message_content: The message content to append images to.
        """
        if isinstance(content, list):
            # Handle new page-based format
            for page in content:
                if isinstance(page, dict) and 'image' in page:
                    self._append_images(page['image'], message_content)
        elif isinstance(content, dict):
            # Handle legacy format
            image_data = content.get('image') or content.get('images')
            self._append_images(image_data, message_content)

    def _append_images(
        self,
        image_data: Union[Dict[str, Any], List[Any], Any],
        message_content: List[Dict[str, Any]],
    ) -> None:
        """
        Append images to the message content.

        Args:
            image_data: The image data to process.
            message_content: The message content to append images to.
        """
        if not image_data:
            return

        if isinstance(image_data, dict):
            images_list = image_data.values()
        elif isinstance(image_data, list):
            images_list = image_data
        else:
            images_list = [image_data]

        for img in images_list:
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
        vision: bool,
    ) -> List[Dict[str, Any]]:
        """
        Build the messages to send to the LLM.

        Args:
            message_content: The content of the message.
            vision: Whether vision is enabled.

        Returns:
            A list of messages.
        """
        system_message = {
            "role": "system",
            "content": "You are a server API that receives document information and returns specific fields in JSON format.",
        }

        if vision:
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

        # No document loader available, check if we can use LLM's vision capabilities
        if not litellm.supports_vision(self.llm.model):
            raise ValueError(
                f"Model {self.llm.model} does not support vision. "
                "Please provide a document loader or a model that supports vision."
            )
    
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
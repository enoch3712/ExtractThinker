import asyncio
import base64
from io import BytesIO
from typing import Any, Dict, List, Optional, IO, Type, Union, get_origin, get_args
from instructor.batch import BatchJob
import uuid
import litellm
from pydantic import BaseModel
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
    encode_image,
    json_to_formatted_string,
    num_tokens_from_string,
)
import yaml
from copy import deepcopy

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

    def extract(
        self,
        source: Union[str, IO, list],
        response_model: type[BaseModel],
        vision: bool = False,
        content: Optional[str] = None,
        completion_strategy: Optional[CompletionStrategy] = None
    ) -> Any:
        """
        Extract information from the provided source.

        Args:
            source (Union[str, IO, list]): The input source which can be a file path, URL, IO stream, or list of dictionaries.
            response_model (type[BaseModel]): The Pydantic model to parse the response into.
            vision (bool, optional): Whether to use vision capabilities. Defaults to False.
            content (Optional[str], optional): Additional content to include in the extraction. Defaults to None.

        Returns:
            Any: The extracted information as specified by the response_model.

        Raises:
            ValueError: If the source type is unsupported or dependencies are not met.
        """
        self._validate_dependencies(response_model, vision)
        self.extra_content = content
        self.completion_strategy = completion_strategy
        # Set vision mode on document loader if available
        if vision and self.document_loader:
            self.document_loader.set_vision_mode(True)

        if vision and not self.get_document_loader(source):
            if not litellm.supports_vision(self.llm.model):
                raise ValueError(
                    f"Model {self.llm.model} does not support vision. Please provide a document loader or a model that supports vision."
                )
            else:
                self.document_loader = DocumentLoaderLLMImage(llm=self.llm)
                self.document_loader.set_vision_mode(True)

        if isinstance(source, str):
            if source.startswith(('http://', 'https://')):
                return self.extract_from_file(source, response_model, vision)
            elif os.path.isfile(source):
                return self.extract_from_file(source, response_model, vision)
            else:
                # Optionally, you can differentiate between invalid paths and actual content
                if any(char in source for char in '/\\'):  # Basic check for path-like strings
                    raise ValueError(f"The file path '{source}' does not exist.")
                return self.extract_from_content(source, response_model, vision)
        elif isinstance(source, list):
            if all(isinstance(item, dict) for item in source):
                return self.extract_from_list(source, response_model, vision)
            else:
                raise ValueError("All items in the source list must be dictionaries.")
        elif hasattr(source, 'read'):
            return self.extract_from_stream(source, response_model, vision)
        else:
            raise ValueError(
                "Source must be a file path, a URL, a stream, or a list of dictionaries."
            )

    async def extract_async(
        self,
        source: Union[str, IO, list],
        response_model: type[BaseModel],
        vision: bool = False,
    ) -> Any:
        return await asyncio.to_thread(self.extract, source, response_model, vision)
    
    def extract_from_content(
        self, content: str, response_model: type[BaseModel], vision: bool = False
    ) -> str:
        return self._extract(content, None, response_model, vision)

    def extract_from_list(
        self, 
        data: List[Dict[Any, Any]], 
        response_model: type[BaseModel], 
        vision: bool
    ) -> str:
        # check if document_loader is None, raise error
        if self.document_loader is None:
            raise ValueError("Document loader is not set")

        # Prepare the structured content
        processed_data = {
            "content": "\n".join(
                f"#{k}:\n{v}"
                for d in data
                for k, v in d.items()
                if k != "image"
            )
        }

        if vision:
            processed_data["images"] = [
                d["image"] for d in data 
                if "image" in d
            ]

        return self._extract(processed_data, response_model, vision) #, is_stream=False)

    def extract_from_file(
        self, file: str, response_model: type[BaseModel], vision: bool = False
    ) -> str:
        if self.document_loader is not None:
            # content = self.document_loader.load_content_from_file(file)
            content = self.document_loader.load(file)
        else:
            document_loader = self.get_document_loader_for_file(file)
            if document_loader is None:
                raise ValueError("No suitable document loader found for file type")
            # content = document_loader.load_content_from_file(file)
            content = self.document_loader.load(file)
        return self._extract(content, response_model, vision)

    def extract_from_stream(
        self, stream: IO, response_model: type[BaseModel], vision: bool = False
    ) -> str:
        # check if document_loader is None, raise error
        if self.document_loader is None:
            raise ValueError("Document loader is not set")

        content = self.document_loader.load(stream)
        return self._extract(content, stream, response_model, vision) #, is_stream=True)

    def classify_from_image(
        self, image: Any, classifications: List[Classification]
    ):
        # requires no content extraction from loader
        content = {
            "image": image,
        }
        return self._classify(content, classifications, image)

    def classify_from_path(
        self, path: str, classifications: List[Classification]
    ):
        content = (
            self.document_loader.load_content_from_file_list(path)
            if self.is_classify_image
            else self.document_loader.load_content_from_file(path)
        )
        return self._classify(content, classifications)

    def classify_from_stream(
        self, stream: IO, classifications: List[Classification]
    ):
        content = (
            self.document_loader.load_content_from_stream_list(stream)
            if self.is_classify_image
            else self.document_loader.load_content_from_stream(stream)
        )
        self._classify(content, classifications)

    def classify_from_excel(
        self, path: Union[str, IO], classifications: List[Classification]
    ):
        if isinstance(path, str):
            content = self.document_loader.load_content_from_file(path)
        else:
            content = self.document_loader.load_content_from_stream(path)
        return self._classify(content, classifications)

    def _add_classification_structure(self, classification: Classification) -> str:
        content = ""
        if classification.contract:
            content = "\tContract Structure:\n"
            # Iterate over the fields of the contract attribute if it's not None
            for name, field in classification.contract.model_fields.items():
                # Extract the type and required status from the field's string representation
                field_str = str(field)
                field_type = field_str.split('=')[1].split(' ')[0]  # Extracts the type
                required = 'required' in field_str  # Checks if 'required' is in the string
                # Creating a string representation of the field attributes
                attributes = f"required={required}"
                # Append each field's details to the content string
                field_details = f"\t\tName: {name}, Type: {field_type}, Attributes: {attributes}"
                content += field_details + "\n"
        return content

    def _classify(
        self, content: Any, classifications: List[Classification], image: Optional[Any] = None
    ):
        messages = [
            {
                "role": "system",
                "content": "You are a server API that receives document information "
                "and returns specific fields in JSON format.\n",
            },
        ]

        if self.is_classify_image:
            input_data = (
                f"##Take the first image, and compare to the several images provided. Then classify according to the classification attached to the image\n"
                + "Output Example: \n"
                + "{\r\n\t\"name\": \"DMV Form\",\r\n\t\"confidence\": 8\r\n}"
                + "\n\n##ClassificationResponse JSON Output\n"
            )

        else:
            input_data = (
                f"##Content\n{content}\n##Classifications\n#if contract present, each field present increase confidence level\n"
                + "\n".join(
                    [
                        f"{c.name}: {c.description} \n{self._add_classification_structure(c)}"
                        for c in classifications
                    ]
                )
                + "#Don't use contract structure, just to help on the ClassificationResponse\nOutput Example: \n"
                + "{\r\n\t\"name\": \"DMV Form\",\r\n\t\"confidence\": 8\r\n}"
                + "\n\n##ClassificationResponse JSON Output\n"
            )

        if self.is_classify_image:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": input_data,
                        },
                    ],
                }
            )
            for classification in classifications:
                if classification.image:
                    if not litellm.supports_vision(model=self.llm.model):
                        raise ValueError(
                            f"Model {self.llm.model} is not supported for vision, since it's not a vision model."
                        )

                    messages.append(
                        {
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
                        }
                    )
                else:
                    raise ValueError(
                        f"Image required for classification '{classification.name}' but not found."
                    )

            response = self.llm.request(messages, ClassificationResponse)
        else:
            messages.append({"role": "user", "content": input_data})
            response = self.llm.request(messages, ClassificationResponse)

        return response

    def classify(
        self,
        input: Union[str, IO],
        classifications: List[Classification],
        image: bool = False,
    ):
        self.is_classify_image = image

        if image:
            return self.classify_from_image(input, classifications)

        document_loader = self.get_document_loader_for_file(input)
        if document_loader is None:
            raise ValueError("No suitable document loader found for the input.")

        content = document_loader.load(input)
        return self._classify(content, classifications)

    async def classify_async(
        self, input: Union[str, IO], classifications: List[Classification]
    ):
        return await asyncio.to_thread(self.classify, input, classifications)

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
                # Prepare content for each source
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

                    messages = [
                        {
                            "role": "system",
                            "content": "You are a server API that receives document information and returns specific fields in JSON format.",
                        },
                        {
                            "role": "user",
                            "content": message_content,
                        },
                    ]
                else:
                    if isinstance(src, str):
                        if os.path.exists(src):
                            content_data = self.document_loader.load_content_from_file(src)
                        else:
                            content_data = src  # Assume src is the text content
                    elif isinstance(src, IO):
                        content_data = self.document_loader.load_content_from_stream(src)
                    else:
                        raise ValueError("Invalid source type.")

                    message_content = f"##Content\n\n{content_data}"
                    if self.extra_content:
                        message_content = f"##Extra Content\n\n{self.extra_content}\n\n" + message_content

                    messages = [
                        {
                            "role": "system",
                            "content": "You are a server API that receives document information and returns specific fields in JSON format.",
                        },
                        {
                            "role": "user",
                            "content": message_content,
                        },
                    ]
                yield messages

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
        """
        Extract information from the content using the LLM.

        Args:
            content: The content to process, which can be a dict, list, or string.
            response_model: The model to format the response.
            vision: Whether to process vision content.

        Returns:
            The response from the LLM.
        """
        # Call all the LLM interceptors before calling the LLM
        for interceptor in self.llm_interceptors:
            interceptor.intercept(self.llm)

        if vision:
            if not litellm.supports_vision(model=self.llm.model):
                raise ValueError(
                    f"Model {self.llm.model} is not supported for vision, since it's not a vision model."
                )

        # Build the message content
        message_content = self._build_message_content(content, vision)

        # Build the messages
        messages = self._build_messages(message_content, vision)

        # Add extra content if it exists
        if self.extra_content is not None:
            self._add_extra_content(messages)

        # Send the request
        response = self.llm.request(messages, response_model)
        return response

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

        Args:
            content: The content to process.

        Returns:
            A string representation of the content.
        """
        if isinstance(content, dict):
            filtered_content = {
                k: v for k, v in content.items()
                if k != 'images' and not hasattr(v, 'read')
            }
            if filtered_content.get("is_spreadsheet", False):
                content_str = json_to_formatted_string(filtered_content.get("data", {}))
            else:
                content_str = yaml.dump(filtered_content, default_flow_style=True)
            return content_str
        return None

    def _add_images_to_message_content(
        self,
        content: Union[Dict[str, Any], List[Any]],
        message_content: List[Dict[str, Any]],
    ) -> None:
        """
        Add images to the message content.

        Args:
            content: The content containing images.
            message_content: The message content to append images to.
        """
        if isinstance(content, list):
            for page in content:
                if isinstance(page, dict) and ('image' in page or 'images' in page):
                    image_data = page.get('image') or page.get('images')
                    self._append_images(image_data, message_content)
        elif isinstance(content, dict):
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

    def _convert_content_to_string(
        self,
        content: Union[Dict[str, Any], List[Any], str]
    ) -> Optional[str]:
        """
        Convert content to a string representation.

        Args:
            content: The content to convert.

        Returns:
            A string representation of the content.
        """
        if isinstance(content, dict):
            if content.get("is_spreadsheet", False):
                return json_to_formatted_string(content.get("data", {}))
            else:
                return yaml.dump(content, default_flow_style=True)
        elif isinstance(content, list):
            return yaml.dump(content, default_flow_style=True)
        elif isinstance(content, str):
            return content
        else:
            return None

    def loadfile(self, file):
        self.file = file
        return self

    def loadstream(self, stream):
        return self
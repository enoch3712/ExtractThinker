import asyncio
from typing import Any, Dict, List, Optional, IO, Type, Union, TypeVar, Generic, cast
import base64
import yaml
from pydantic import BaseModel, Field
from extract_thinker.document_loader.document_loader import DocumentLoader
from extract_thinker.llm import LLM
from concurrent.futures import ThreadPoolExecutor, as_completed
from extract_thinker.utils import encode_image, json_to_formatted_string, extract_thinking_json
import re
from io import BytesIO
from uuid import uuid4
from PIL import Image

class ContentItem(BaseModel):
    """Represents a single piece of extracted content with certainty."""
    certainty: int = Field(..., ge=1, le=10, description="Confidence score (1-10) for the extracted content.")
    content: str = Field(..., description="The extracted text content or description.")

class PageContent(BaseModel):
    """Structured content extracted from a single page."""
    items: List[ContentItem] = Field(..., description="List of content items extracted from the page.")

# --- Markdown Converter Class ---

class MarkdownConverter:
    """
    Converts documents (text, images, etc.) into Markdown format, potentially
    using an LLM for structured content extraction first.
    Supports a `vision` flag to control image processing.
    Uses message building logic copied from Extractor class for consistency.
    """

    DEFAULT_PAGE_PROMPT = """
First, convert the image into well-formatted Markdown content.
After your Markdown content, provide a JSON representation of the content in the following format:

JSON Schema:
{
  "items": [
    {
      "certainty": number (1-10),
      "content": "string containing a section of the markdown content"
    }
  ]
}

Instructions:
1. Create proper Markdown with headings, lists, formatting etc.
   - Format headings with # syntax (# for main headings, ## for sub-headings, etc.)
   - Format lists with proper bullet points (*, -) or numbers (1., 2., etc.)
   - Apply proper emphasis using **bold**, *italic*, or `code` where appropriate
   - Create proper links using [text](url) format if applicable
   - Format code blocks with triple backticks ``` if applicable
   - Format tables using proper Markdown table syntax if applicable
   - Use block quotes with > where appropriate
2. Include ALL content from the image in your Markdown output.
3. After the Markdown section, add a JSON block that follows the above schema.
4. The JSON should break down the Markdown into logical sections with certainty scores.
5. Make sure the certainty scores accurately reflect your confidence (1-10).
6. Make sure you keep things like checkboxes, lists, etc. as is.
7. IMPORTANT: If you see a placeholder for an image (e.g. [img-1.jpeg], <!-- image -->, etc.), NEVER include the placeholder directly in your output. ALWAYS replace it with the actual content or description of what the image shows.
   Example: Replace [img-1.jpeg] with "Signature of William Smith" or appropriate description of what's in the image.
8. MAINTAIN ALL THE ORIGINAL CONTENT AND MEANING - do not add or remove information.

Your response format should be:

<MARKDOWN CONTENT>
All the formatted content here...

```json
{
  "items": [
    {
      "certainty": 10,
      "content": "## Some Title"
    },
    {
      "certainty": 8,
      "content": "### Some content"
    }
  ]
}
```

Focus on creating high-quality, well-structured Markdown first, then provide the JSON breakdown.
"""

    DEFAULT_MARKDOWN_PROMPT = """
Convert the image into well-formatted Markdown content.

Instructions:
Create proper Markdown with headings, lists, formatting etc.
Include ALL content from the image in your Markdown output.
Format headings with # syntax (# for main headings, ## for sub-headings, etc.)
Format lists with proper bullet points (*, -) or numbers (1., 2., etc.)
Apply proper emphasis using **bold**, *italic*, or `code` where appropriate
Create proper links using [text](url) format if applicable
Format code blocks with triple backticks ``` if applicable
Format tables using proper Markdown table syntax if applicable
Make sure you keep things like checkboxes, lists, etc. as is.
IMPORTANT: If you see a placeholder for an image (e.g. [img-1.jpeg], <!-- image -->, etc.), NEVER include the placeholder directly in your output. ALWAYS replace it with the actual content or description of what the image shows.
Example: Replace [img-1.jpeg] with "Signature of William Smith" or appropriate description of what's in the image.

Your response should ONLY include the formatted Markdown content without any additional JSON structure or explanations.
"""

    PLACEHOLDER_INSTRUCTION = (
        "**Formatting Instruction:** The text above contains placeholders "
        "(like '[Image content not extracted...]') indicating where visual information "
        "from an image could not be retrieved."
    )

    def __init__(
        self, document_loader: Optional[DocumentLoader] = None, llm: Optional[LLM] = None
    ):
        self.document_loader: Optional[DocumentLoader] = document_loader
        self.llm: Optional[LLM] = llm
        self.allow_vision = False
        self._allow_verification = False

    @property
    def allow_verification(self) -> bool:
        """Gets the verification flag."""
        return self._allow_verification
    
    @allow_verification.setter
    def allow_verification(self, value: bool) -> None:
        """Sets the verification flag."""
        self._allow_verification = value

    def load_document_loader(self, document_loader: DocumentLoader) -> None:
        self.document_loader = document_loader

    def load_llm(self, llm: LLM) -> None:
        self.llm = llm

    def _validate_dependencies(self, require_llm: bool = False) -> None:
        """
        Validates that required dependencies are present.
        """
        if self.document_loader is None:
            raise ValueError("Document loader is not set. Please set a document loader.")
        if require_llm and self.llm is None:
            raise ValueError("LLM is required for this operation but not set.")

    def to_markdown_structured(self, source: Union[str, IO, List[Union[str, IO]]], pages: Optional[List[int]] = None) -> List[PageContent]:
        """
        Processes document(s) using the LLM to extract structured content per page.
        This method requires vision capabilities and expects the document to contain images.
        
        Args:
            source: A single file path/stream or a list of them.
            pages: Optional list of page numbers to process (1-indexed). If None, all pages are processed.

        Returns:
            A list where each element is a string containing Markdown content followed by 
            a JSON structure (for successful pages) or an error message (for failed pages).

        Raises:
            ValueError: If the document loader does not find any images in the source.
            ValueError: If any specified page number is invalid.
        """
        self._validate_dependencies(require_llm=True) # LLM is mandatory here
        if pages == []:
            return []

        if isinstance(source, list):
            # TODO: Implement handling for multiple sources if needed
            raise NotImplementedError("Handling multiple sources is not yet implemented.")
        else:
            # Handle single source
            if not self.document_loader:
                 raise ValueError("Document loader is required.") # Redundant check

            # Configure document loader for vision - this method requires it.
            if hasattr(self.document_loader, 'set_vision_mode'):
                try:
                    self.document_loader.set_vision_mode(True)
                except Exception as e:
                    # If setting vision mode fails, we probably can't proceed as expected.
                    raise ValueError(f"Failed to set vision mode on document loader: {e}") from e
            else:
                print("Warning: Document loader does not have set_vision_mode. Assuming it handles vision implicitly.")

            pages_data = self.document_loader.load(source)
            if not isinstance(pages_data, list):
                pages_data = [pages_data] if pages_data else []

            # Check if any images were actually loaded
            has_images = any(isinstance(page, dict) and page.get('image') for page in pages_data)
            if not has_images:
                raise ValueError("to_markdown_structured requires a document containing images, but none were found by the loader.")

            pages_data = self._select_pages(pages_data, pages)

            result_strings = [None] * len(pages_data) # Pre-allocate list

            with ThreadPoolExecutor() as executor:
                future_to_index = {executor.submit(self._process_page_with_llm, page_data): i
                                   for i, page_data in enumerate(pages_data)}

                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        result_strings[index] = future.result()
                    except Exception as exc:
                        raise ValueError(f"Structured Markdown conversion failed for page {index + 1}") from exc

            return result_strings
        
    def _build_messages(
        self,
        message_content: Union[List[Dict[str, Any]], List[str]],
        structured: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Build the messages to send to the LLM.

        Args:
            message_content: The content of the message.
            structured: Whether to request structured content with JSON (True) or just Markdown (False).

        Returns:
            A list of messages.
        """
        system_message = {
            "role": "system",
            "content": self.DEFAULT_PAGE_PROMPT if structured else self.DEFAULT_MARKDOWN_PROMPT
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
    
    def _build_message_content(
        self,
        content: Optional[Union[Dict[str, Any], List[Any], str]],
        vision: bool,
    ) -> Union[List[Dict[str, Any]], List[str]]:
        """
        Build the message content based on the content and vision flag.
        If vision=True and placeholders are detected, modifies the content
        sent to the LLM to include instructions for handling them.

        Args:
            content: The content to process.
            vision: Whether to process vision content.

        Returns:
            A list representing the message content.
        """
        message_content: Union[List[Dict[str, Any]], List[str]] = []

        if content is None:
            return message_content

        processed_text = None
        if vision:
            # Add images first (if any)
            self._add_images_to_message_content(content, message_content)
            # Process text content separately
            processed_text = self._process_content_data(content)
        else:
            # Process non-vision content to string
            processed_text = self._convert_content_to_string(content)

        # Prepare the final text to be sent to the LLM
        final_text_for_llm = processed_text if processed_text else ""

        # --- Modification for Placeholders when Vision is Active ---
        if vision and processed_text:
            placeholder_pattern = r"\[(?:Image content not extracted|Image processing failed|Image content extraction failed).*?\]"
            # Find all occurrences of placeholders
            placeholders_found = re.findall(placeholder_pattern, processed_text)

            if placeholders_found:
                print(f"Detected {len(placeholders_found)} image extraction placeholder(s). Adding specific instructions to content.")
                # Construct an instruction to append *after* the main text
                instruction = (
                    "\n\n---\n"
                    + self.PLACEHOLDER_INSTRUCTION + "\n---\n"
                )
                # Append the instruction to the text that will be sent
                final_text_for_llm += instruction
        # --- End Modification ---


        # Construct the final message content block
        # Use the potentially modified final_text_for_llm
        final_text_content_block = "## Content\n\n" + (final_text_for_llm if final_text_for_llm else "[No text content available]")

        if vision:
             # Append as a text block if vision was used (images added earlier)
             # Make sure message_content is not empty if only images were present initially
             if not message_content and not final_text_for_llm:
                 # Only images were present, but maybe they failed? Or no images?
                 final_text_content_block = "## Content\n\n[No processable text or image content found for this page]"

             message_content.append({
                 "type": "text",
                 "text": final_text_content_block
             })
        else:
             # For non-vision, the message content is just the string list
             message_content = [final_text_content_block]


        # Final check for empty vision content (e.g., only images failed)
        # This logic seems less critical now as we handle it above, but keep for safety
        if vision and not any(item.get("text", "").strip() for item in message_content if isinstance(item, dict)) \
           and not any(item.get("type") == "image_url" for item in message_content if isinstance(item, dict)):
             # If truly empty after all processing
             message_content = [{
                 "type": "text",
                 "text": "## Content\n\n[Empty page or only content that failed processing]"
             }]


        return message_content

    def _process_page_with_llm(self, page_data: Any) -> PageContent:
        """
        Uses the LLM to process a single page's data and return the raw response.
        Always processes images if they are present in page_data.

        Args:
            page_data: The data for a single page (expected dict with 'content', 'images').

        Returns:
            A string containing the raw LLM response with Markdown and JSON sections.
        """
        self.allow_vision = True

        if not self.llm:
            raise ValueError("LLM is required for structured extraction but not configured.")

        if not isinstance(page_data, dict):
            raise ValueError("Expected a page dictionary for structured Markdown conversion")

        messages = self._build_messages(self._build_message_content(page_data, vision=True))

        try:
            # Use raw_completion to get the full text response instead of parsing it into a model
            raw_response = self.llm.raw_completion(messages=messages)
            return extract_thinking_json(raw_response, PageContent)
        except Exception as e:
            print(f"LLM request failed for page: {e}")
            raise

    def _process_content_data(
        self,
        content: Union[Dict[str, Any], List[Any], str],
    ) -> Optional[str]:
        """
        Process content data by filtering out images and converting to a string.
        (Copied from Extractor class)
        Args:
            content: The content to process.

        Returns:
            A string representation of the content.
        """
        if isinstance(content, list):
            # This case might not be typical if called with single page_data
            page_texts = []
            for page in content:
                if isinstance(page, dict):
                    page_text = page.get('content', '')
                    if page_text:
                        page_texts.append(page_text)
            return "\n\n".join(page_texts) if page_texts else None

        elif isinstance(content, dict):
            # Main path when called with page_data
            filtered_content = {
                k: v for k, v in content.items()
                if k != 'images' and k != 'image' and not hasattr(v, 'read')
            }
            is_spreadsheet = content.get("is_spreadsheet")
            if is_spreadsheet is False:
                # Handle potential spreadsheet data if loader adds this flag
                content_str = json_to_formatted_string(filtered_content.get("data", {}))
            else:
                content_str = json_to_formatted_string(filtered_content)
                
                # remove 'is_spreadsheet: true' content from the string
                content_str = content_str.replace('is_spreadsheet: true', '')
            return content_str
        elif isinstance(content, str):
            return content
        return None

    # Added this method as it is used by _build_message_content when vision=False
    def _convert_content_to_string(
        self,
        content: Union[Dict[str, Any], List[Any], str]
    ) -> Optional[str]:
        """
        Convert content to a string representation.
        (Copied from Extractor class)
        Args:
            content: The content to convert.

        Returns:
            A string representation of the content.
        """
        if isinstance(content, list):
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
                # Prefer 'content' key, otherwise dump YAML
                content_str = content.get('content')
                return content_str if content_str is not None else yaml.dump(content, default_flow_style=True)
        elif isinstance(content, str):
            return content
        return None

    def _add_images_to_message_content(
        self,
        content: Union[Dict[str, Any], List[Any]],
        message_content: List[Dict[str, Any]],
    ) -> None:
        """
        (Copied from Extractor class)
        Recursively adds images found in content to message_content list.
        """
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
        (Copied from Extractor class)
        Append images to the message content.

        Args:
            image_data: The image data to process. Can be dict, list, or single image.
            message_content: The message content list to append images to.
        """
        if not image_data:
            return

        images_list = []
        if isinstance(image_data, dict):
            # Handle dictionary format (potentially from Mistral containing base64)
            if "images" in image_data and isinstance(image_data["images"], list):
                images_list.extend(image_data["images"])
            elif "image" in image_data and image_data["image"] is not None:
                images_list.append(image_data["image"])
            # Special case: if the dict itself IS the image data (like from Mistral)
            elif 'base64' in image_data: # Check if it looks like the Mistral image dict
                 images_list.append(image_data) # Append the dict itself
        elif isinstance(image_data, list):
            # Process list of images or image dictionaries
            images_list.extend(image_data) # Directly extend if it's already a list
        else:
            # Single raw image (bytes, path, etc.)
            images_list.append(image_data)

        # Process all collected images
        for img in images_list:
            if img is not None:  # Skip None values
                base64_image = None
                try:
                    if isinstance(img, dict) and 'base64' in img:
                         # Handle Mistral-like dictionary containing base64 string
                         base64_payload = img.get('base64')
                         if base64_payload.startswith('data:'):
                              base64_image = base64_payload.split(',', 1)[-1]
                         else:
                              base64_image = base64_payload
                    else:
                         # Use encode_image for bytes, paths, etc.
                         base64_image = encode_image(img)

                    if base64_image:
                        message_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        })
                    else:
                         print(f"Warning: Could not get base64 for image item: {type(img)}")
                except Exception as e:
                     print(f"Warning: Error processing image item {type(img)}: {e}")

    # --- Copied Methods from Extractor --- END ---

    @staticmethod
    def _select_pages(pages_data: List[Any], pages: Optional[List[int]]) -> List[Any]:
        if pages is None:
            return pages_data
        if any(isinstance(number, bool) or not isinstance(number, int) or number < 1 for number in pages):
            raise ValueError("Page numbers must be positive integers")
        if len(set(pages)) != len(pages):
            raise ValueError("Page numbers must not contain duplicates")
        if pages and max(pages) > len(pages_data):
            raise ValueError(f"Page number {max(pages)} exceeds total number of pages ({len(pages_data)})")
        return [pages_data[number - 1] for number in pages]

    def to_markdown(self, source: Union[str, IO, List[Union[str, IO]]], vision: bool = False,
                    pages: Optional[List[int]] = None, *, preserve_tags: bool = False,
                    include_images: bool = False) -> List[str]:
        """Convert each page to Markdown, optionally retaining tags and page images.

        `vision` controls sending images to the model. `include_images` embeds
        original images in the returned Markdown independently of model vision.
        With `preserve_tags`, source HTML/XML tags are protected during conversion;
        missing or reordered markers cause an error instead of losing the tags.
        """
        if not self.llm:
            raise ValueError("LLM is required for markdown conversion but not configured.")
        if pages == []:
            return []
        if self.document_loader:
            self.document_loader.set_vision_mode(vision or include_images)
            pages_data = []
            sources = source if isinstance(source, list) else [source]
            for item in sources:
                loaded = self.document_loader.load(item)
                pages_data.extend(loaded if isinstance(loaded, list) else [loaded])
        else:
            if isinstance(source, list) or vision or include_images:
                raise ValueError("A document loader is required for multiple sources or images")
            if isinstance(source, str):
                with open(source, encoding="utf-8") as file:
                    text = file.read()
            elif hasattr(source, "read"):
                text = source.read()
                if isinstance(text, bytes):
                    text = text.decode("utf-8")
            else:
                raise ValueError("Source must be a file path or readable stream")
            pages_data = [{"content": text}]
        # Use the original page index in embedded image labels, even after selection.
        indexed = list(enumerate(pages_data, start=1))
        indexed = self._select_pages(indexed, pages)
        if not indexed:
            return []
        output = [""] * len(indexed)
        with ThreadPoolExecutor(max_workers=min(8, len(indexed))) as executor:
            pending = {
                executor.submit(self._process_markdown_page, page, vision, preserve_tags,
                                include_images, number): index
                for index, (number, page) in enumerate(indexed)
            }
            for future in as_completed(pending):
                index = pending[future]
                try:
                    output[index] = future.result()
                except Exception as exc:
                    for task in pending:
                        task.cancel()
                    raise ValueError(f"Markdown conversion failed for page {indexed[index][0]}") from exc
        return output

    def _process_markdown_page(self, page_data: Any, vision: bool = True,
                               preserve_tags: bool = False, include_images: bool = False,
                               page_number: int = 1) -> str:
        if not self.llm:
            raise ValueError("LLM is required for markdown conversion")
        if not isinstance(page_data, dict):
            raise ValueError("Expected a page dictionary")
        working_page = dict(page_data)
        tags = {}
        if preserve_tags:
            text = working_page.pop("markdown", None)
            if text is None:
                text = working_page.get("content", "")
            prefix = "ET_TAG_" + uuid4().hex + "_"
            def protect(match):
                marker = prefix + str(len(tags)) + "_END"
                tags[marker] = match.group(0)
                return marker
            tag_pattern = r'''<!--.*?-->|<\?.*?\?>|<![A-Za-z][^>]*>|</?[A-Za-z](?:[^>"']|"[^"]*"|'[^']*')*>'''
            working_page["content"] = re.sub(tag_pattern, protect, text, flags=re.DOTALL)
        parts = self._build_message_content(working_page, vision=vision)
        prompt = self.DEFAULT_MARKDOWN_PROMPT
        if preserve_tags:
            prompt += "\nKeep every ET_TAG marker exactly once and in its original order. Do not modify or interpret markers."
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": parts if vision else "".join(parts)},
        ]
        result = self.llm.raw_completion(messages=messages)
        if not isinstance(result, str):
            raise ValueError("Markdown completion must return text")
        positions = []
        for marker in tags:
            if result.count(marker) != 1:
                raise ValueError("Model did not preserve all source tags")
            positions.append(result.index(marker))
        if positions != sorted(positions):
            raise ValueError("Model reordered source tags")
        for marker, tag in tags.items():
            result = result.replace(marker, tag)
        if include_images:
            image_markdown = self._page_images_markdown(page_data, page_number)
            if image_markdown:
                result += "\n\n" + "\n\n".join(image_markdown)
        return result

    @staticmethod
    def _page_images_markdown(page: Dict[str, Any], page_number: int) -> List[str]:
        images = page.get("images") or []
        images = list(images) if isinstance(images, list) else [images]
        if page.get("image") is not None:
            images.append(page["image"])
        output = []
        seen = set()
        for source in images:
            if isinstance(source, dict):
                encoded = source.get("base64")
                if not isinstance(encoded, str):
                    raise ValueError("Image dictionary must contain a base64 string")
                encoded = encoded.split(",", 1)[-1] if encoded.startswith("data:") else encoded
                data = base64.b64decode(encoded, validate=True)
            else:
                data = base64.b64decode(encode_image(source), validate=True)
            if data in seen:
                continue
            seen.add(data)
            with Image.open(BytesIO(data)) as image:
                mime = Image.MIME.get(image.format)
                if mime is None:
                    raise ValueError("Unsupported embedded image format")
            encoded = base64.b64encode(data).decode("ascii")
            output.append(f"![Page {page_number} image {len(output) + 1}](data:{mime};base64,{encoded})")
        return output

    def _basic_to_markdown(self, source: Union[str, IO, List[Union[str, IO]]], vision: bool = False, pages: Optional[List[int]] = None) -> str:
         """
         Performs basic Markdown conversion without using an LLM.
         (Essentially the previous implementation of to_markdown)

         Args:
             source: Input source.
             vision: If True, includes image data in the output.
             pages: Optional list of page numbers to process (1-indexed). If None, all pages are processed.
         """
         self._validate_dependencies(require_llm=False)

         if isinstance(source, list):
             raise NotImplementedError("Basic handling multiple sources is not yet implemented.")
         else:
            if not self.document_loader:
                 raise ValueError("Document loader is required.")

            # Configure document loader for vision if needed and supported
            if hasattr(self.document_loader, 'set_vision_mode'):
                try:
                    self.document_loader.set_vision_mode(vision)
                except Exception as e:
                    print(f"Warning: Failed to set vision mode on document loader: {e}")

            pages_data = self.document_loader.load(source)
            if not isinstance(pages_data, list):
                pages_data = [pages_data] if pages_data else []

            # Validate page numbers if specified
            if pages is not None:
                if not all(isinstance(p, int) and p > 0 for p in pages):
                    raise ValueError("Page numbers must be positive integers")
                if max(pages) > len(pages_data):
                    raise ValueError(f"Page number {max(pages)} exceeds total number of pages ({len(pages_data)})")
                # Convert to 0-based indices
                page_indices = [p - 1 for p in pages]
                pages_data = [pages_data[i] for i in page_indices]

            markdown_parts = [""] * len(pages_data)
            with ThreadPoolExecutor() as executor:
                # Pass vision flag to basic page converter
                future_to_index = {executor.submit(self._convert_page_basic, page_data, vision): i
                                   for i, page_data in enumerate(pages_data)}

                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        markdown_parts[index] = future.result()
                    except Exception as exc:
                        print(f'Basic Page {index + 1} conversion failed: {exc}')
                        markdown_parts[index] = f"\n\n<!-- Error converting page {index + 1}: {exc} -->\n\n"

            return "\n\n".join(part for part in markdown_parts if part)

    def _convert_page_basic(self, page_data: Any, vision: bool) -> str:
        """
        (Previously _convert_page_to_markdown)
        Converts a single page's data to a basic Markdown string (text + image).

        Args:
            page_data: Data for the page.
            vision: If True, include the first image in Markdown format.
        """
        if not isinstance(page_data, dict):
            print(f"Warning: Unexpected page data type: {type(page_data)}. Converting to string.")
            return str(page_data)

        text_content = page_data.get("content", "")
        image_md = ""

        # Only process images if vision is enabled
        if vision:
            images = page_data.get("images", [])
            if images and isinstance(images, list) and len(images) > 0:
                 try:
                     first_image_data = images[0]
                     if isinstance(first_image_data, bytes):
                          b64_img = base64.b64encode(first_image_data).decode('utf-8')
                          # Basic image tag, assuming PNG
                          image_md = f"\n![Page Image](data:image/png;base64,{b64_img})\n"
                     else:
                         print(f"Warning: Image data is not in bytes format: {type(first_image_data)}")
                 except Exception as e:
                     print(f"Error processing image on page: {e}")
                     image_md = "\n<!-- Error processing image -->\n"

        separator = "\n" if text_content and image_md else ""
        return text_content + separator + image_md

    async def to_markdown_structured_async(self, source: Union[str, IO, List[Union[str, IO]]], pages: Optional[List[int]] = None) -> List[PageContent]:
         """ 
         Asynchronously extracts structured content using vision and returns raw strings with
         markdown and JSON sections.
         """
         # No vision flag needed for the sync method call
         return await asyncio.to_thread(self.to_markdown_structured, source, pages=pages)

    async def to_markdown_async(self, source: Union[str, IO, List[Union[str, IO]]], vision: bool = False,
                                pages: Optional[List[int]] = None, *, preserve_tags: bool = False,
                                include_images: bool = False) -> List[str]:
        """Asynchronously convert Markdown with the same options as to_markdown."""
        return await asyncio.to_thread(self.to_markdown, source, vision=vision, pages=pages,
                                       preserve_tags=preserve_tags, include_images=include_images)

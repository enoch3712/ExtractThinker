# Markdown Conversion

The `MarkdownConverter` class provides functionality to convert documents (including text and images) into Markdown format. It leverages a configured Language Model (LLM) for sophisticated conversion, especially when dealing with images or requiring structured output.

## Core Concepts

- **LLM Integration:** The converter **requires** a configured LLM (`extract_thinker.llm.LLM`) to interpret document content and generate well-formatted Markdown. This is essential for both text and vision-based tasks (processing images) and for generating structured JSON output alongside Markdown.
- **Document Loader:** It relies on a `DocumentLoader` (`extract_thinker.document_loader.DocumentLoader`) to load the source document(s) and potentially extract text and images. The behavior might vary depending on the specific loader used.
- **Vision Support:** The `to_markdown` and `to_markdown_structured` methods have a `vision` parameter or operate in vision mode by default. When enabled, the converter attempts to process images within the document using the LLM's vision capabilities (if the LLM supports it).
- **Structured Output:** The `to_markdown_structured` method specifically instructs the LLM to provide not only the Markdown content but also a JSON structure breaking down the content with certainty scores. This method inherently requires vision capabilities in the LLM.

## Initialization

```python
from extract_thinker.markdown import MarkdownConverter
from extract_thinker.document_loader import DocumentLoaderPyPdf # Example loader
from extract_thinker.llm import LLM
from extract_thinker.global_models import get_lite_model, get_big_model # Helpers for model config

# Initialize with or without components
markdown_converter = MarkdownConverter()

# Load components later
loader = DocumentLoaderPyPdf() # Configure as needed
# Use helper functions to get model configurations
# Replace with your actual logic for selecting/configuring models if needed
llm = LLM(get_lite_model()) 

markdown_converter.load_document_loader(loader)
markdown_converter.load_llm(llm)

# Or initialize directly
markdown_converter = MarkdownConverter(document_loader=loader, llm=llm)
```

## Usage

### Simple Markdown Conversion (LLM Required)

This method uses the configured LLM to generate Markdown. If `vision=True`, it processes images (requires an LLM with vision capabilities). **Note:** An LLM must be configured via `load_llm()` or during initialization for this method to work.

```python
# Assuming markdown_converter is initialized with loader and LLM
source_path = "path/to/your/document.pdf" # Or image file like .png, .jpg

# Convert with vision disabled (processes text only using LLM)
markdown_pages_text = markdown_converter.to_markdown(source_path, vision=False) 
# Returns List[str]

# Convert with vision enabled (processes text and images using LLM)
markdown_pages_vision = markdown_converter.to_markdown(source_path, vision=True) 
# Returns List[str]

for i, page_md in enumerate(markdown_pages_vision):
    print(f"--- Page {i+1} ---")
    print(page_md)

# Async version
markdown_pages_vision_async = await markdown_converter.to_markdown_async(source_path, vision=True)
```

### Structured Markdown Conversion (LLM Vision Required)

This method *requires* an LLM with vision capabilities and a document containing images. It returns structured data including Markdown and a JSON breakdown. **Note:** An LLM must be configured via `load_llm()` or during initialization for this method to work.

```python
from extract_thinker.markdown import PageContent

# Assuming markdown_converter is initialized with loader and LLM (with vision)
image_path = "path/to/your/image.png" 

try:
    # This method inherently uses vision
    structured_output: List[PageContent] = markdown_converter.to_markdown_structured(image_path)
    # Returns List[PageContent]

    for i, page_content in enumerate(structured_output):
        print(f"--- Page {i+1} ---")
        # Access structured items
        for item in page_content.items:
            print(f"Certainty: {item.certainty}, Content: {item.content[:50]}...") # Print snippet

except ValueError as e:
    print(f"Error: {e}") # e.g., if no images found or LLM not set

# Async version
structured_output_async: List[PageContent] = await markdown_converter.to_markdown_structured_async(image_path)

```
**Note:** The `to_markdown_structured` method expects the LLM to return both Markdown and a specific JSON format. The `extract_thinking_json` utility is used internally to parse this.

## Prompts

The converter uses specific system prompts depending on the method called:
- `DEFAULT_PAGE_PROMPT`: Used by `to_markdown_structured`. Instructs the LLM to output Markdown *and* a JSON structure.
- `DEFAULT_MARKDOWN_PROMPT`: Used by `to_markdown` (when using LLM). Instructs the LLM to output *only* well-formatted Markdown.
- `MARKDOWN_VERIFICATION_PROMPT`: Potentially used for refining existing text (internal flag `allow_verification`).

These prompts guide the LLM's output format. 
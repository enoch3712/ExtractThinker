import json
import pytest
import os
import sys
from dotenv import load_dotenv
from io import BytesIO

# Ensure the package root is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from extract_thinker.markdown.markdown_converter import MarkdownConverter, PageContent, ContentItem
from extract_thinker.document_loader.document_loader_pypdf import DocumentLoaderPyPdf
from extract_thinker.document_loader.document_loader_spreadsheet import DocumentLoaderSpreadSheet
from extract_thinker.document_loader.document_loader_mistral_ocr import DocumentLoaderMistralOCR, MistralOCRConfig
from extract_thinker.llm import LLM
from extract_thinker.exceptions import ExtractThinkerError # Assuming common exceptions if needed
from extract_thinker.global_models import get_lite_model, get_big_model

load_dotenv()
cwd = os.getcwd()

# --- Test File Paths ---
PDF_PATH = os.path.join(cwd, "tests", "files", "invoice.pdf")
IMAGE_PATH = os.path.join(cwd, "tests", "test_images", "invoice.png")
SPREADSHEET_PATH = os.path.join(cwd, "tests", "files", "family_budget.xlsx")

# --- Helper Functions (to avoid repetition) ---
def get_mistral_config():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return None
    return MistralOCRConfig(api_key=api_key)

def test_pypdf_basic_conversion_no_vision():
    """Test basic Markdown conversion (no LLM) with PyPDF, vision off."""
    loader = DocumentLoaderPyPdf()
    converter = MarkdownConverter(document_loader=loader)
    markdown_output = converter.to_markdown(PDF_PATH, vision=False)

    assert isinstance(markdown_output, str)
    assert len(markdown_output) > 0
    assert "Invoice" in markdown_output # Basic check for text content
    assert "![Page Image]" not in markdown_output # Should not contain image tag

def test_pypdf_basic_conversion_with_vision():
    """Test basic Markdown conversion (no LLM) with PyPDF, vision on."""
    loader = DocumentLoaderPyPdf()
    converter = MarkdownConverter(document_loader=loader)
    markdown_output = converter.to_markdown(PDF_PATH, vision=True)

    assert isinstance(markdown_output, str)
    assert len(markdown_output) > 0
    assert "Invoice" in markdown_output
    # Image tag format might vary slightly, check presence
    assert "![Page Image](data:image/png;base64," in markdown_output

def test_pypdf_llm_conversion_no_vision():
    """Test LLM-based Markdown conversion with PyPDF, vision off."""
    loader = DocumentLoaderPyPdf()
    llm = LLM(get_lite_model())
    converter = MarkdownConverter(document_loader=loader, llm=llm)
    markdown_output = converter.to_markdown(PDF_PATH, vision=False)

    assert isinstance(markdown_output, str)
    assert len(markdown_output) > 0
    # LLM output is less predictable, check for basic characteristics
    assert "Invoice" in markdown_output # Should still contain core text

@pytest.mark.slow # Mark as slow because it uses an LLM API call
def test_pypdf_llm_conversion_with_vision():
    """Test LLM-based Markdown conversion with PyPDF, vision on."""
    loader = DocumentLoaderPyPdf()
    llm = LLM(get_lite_model())
    converter = MarkdownConverter(document_loader=loader, llm=llm)
    markdown_output = converter.to_markdown(PDF_PATH, vision=True)

    assert isinstance(markdown_output, str)
    assert len(markdown_output) > 0
    # Check that LLM processed something related to the content
    assert "Invoice" in markdown_output or "Bill To" in markdown_output

def test_pypdf_structured_conversion_no_vision():
    """Test structured conversion with PyPDF, vision off."""
    loader = DocumentLoaderPyPdf()
    llm = LLM(get_lite_model())
    converter = MarkdownConverter(document_loader=loader, llm=llm)
    results = converter.to_markdown_structured(PDF_PATH, vision=False)

    assert isinstance(results, list)
    assert len(results) == 1 # Expecting 1 page for invoice.pdf
    assert isinstance(results[0], PageContent)
    assert len(results[0].items) > 0
    assert isinstance(results[0].items[0], ContentItem)
    assert isinstance(results[0].items[0].certainty, int)
    assert isinstance(results[0].items[0].content, str)

@pytest.mark.slow # Mark as slow because it uses an LLM API call
def test_pypdf_structured_conversion_with_vision():
    """Test structured conversion with PyPDF, vision on."""
    loader = DocumentLoaderPyPdf()
    llm = LLM(get_lite_model())
    converter = MarkdownConverter(document_loader=loader, llm=llm)
    results = converter.to_markdown_structured(PDF_PATH, vision=True)

    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], PageContent)
    assert len(results[0].items) > 0
    assert isinstance(results[0].items[0], ContentItem)

# === Spreadsheet Loader Tests ===

def test_spreadsheet_basic_conversion():
    """Test basic Markdown conversion (no LLM) for spreadsheet."""
    try:
        loader = DocumentLoaderSpreadSheet()
    except ImportError as e:
        pytest.skip(f"Skipping spreadsheet test: {e}")

    # Vision flag doesn't apply to basic spreadsheet conversion's content part
    converter = MarkdownConverter(document_loader=loader)
    markdown_output = converter.to_markdown(SPREADSHEET_PATH, vision=False)

    assert isinstance(markdown_output, str)
    assert len(markdown_output) > 0
    # Check for content expected from family_budget.xlsx
    assert "Monthly Income" in markdown_output
    assert "Salary" in markdown_output
    assert "Housing" in markdown_output
    assert "![Page Image]" not in markdown_output # Basic conversion doesn't add images

@pytest.mark.slow # Mark as slow because it uses an LLM API call
def test_spreadsheet_llm_conversion():
    """Test LLM-based Markdown conversion for spreadsheet."""
    try:
        loader = DocumentLoaderSpreadSheet()
    except ImportError as e:
        pytest.skip(f"Skipping spreadsheet test: {e}")

    llm = LLM(get_lite_model())
    converter = MarkdownConverter(document_loader=loader, llm=llm)
    # Vision is False as spreadsheet loader doesn't provide images to LLM this way
    markdown_output = converter.to_markdown(SPREADSHEET_PATH, vision=False)

    assert isinstance(markdown_output, str)
    assert len(markdown_output) > 0
    assert "Monthly Income" in markdown_output or "Salary" in markdown_output

@pytest.mark.slow # Mark as slow because it uses an LLM API call
def test_spreadsheet_structured_conversion():
    """Test structured conversion for spreadsheet."""
    try:
        loader = DocumentLoaderSpreadSheet()
    except ImportError as e:
        pytest.skip(f"Skipping spreadsheet test: {e}")

    llm = LLM(get_lite_model())
    converter = MarkdownConverter(document_loader=loader, llm=llm)
    results = converter.to_markdown_structured(SPREADSHEET_PATH, vision=False)

    assert isinstance(results, list)
    assert len(results) > 0 # Should have multiple sheets/pages
    # Check first sheet/page
    assert isinstance(results[0], PageContent)
    assert len(results[0].items) > 0
    assert isinstance(results[0].items[0], ContentItem)
    # Check content indicative of spreadsheet data
    found_budget_term = any("budget" in item.content.lower() or "income" in item.content.lower() for item in results[0].items)
    assert found_budget_term

# === MistralOCR Loader Tests ===
# Note: These rely on MISTRAL_API_KEY environment variable

@pytest.mark.slow # Mark as slow due to external API call
def test_mistral_llm_conversion_image():
    """Test LLM-based Markdown conversion with Mistral OCR on an image."""
    mistral_config = get_mistral_config()
    if not mistral_config:
        pytest.skip("Skipping Mistral OCR test: MISTRAL_API_KEY not set.")

    loader = DocumentLoaderMistralOCR(config=mistral_config)
    llm = LLM(get_lite_model())
    converter = MarkdownConverter(document_loader=loader, llm=llm)
    # Mistral loader implies vision=True usage naturally
    markdown_output = converter.to_markdown(IMAGE_PATH, vision=True)

    assert isinstance(markdown_output, str)
    assert len(markdown_output) > 0
    # Check for content expected from the image OCR
    assert "Invoice" in markdown_output or "Example Corp" in markdown_output

@pytest.mark.slow # Mark as slow due to external API call
def test_mistral_llm_conversion_pdf():
    """Test LLM-based Markdown conversion with Mistral OCR on a PDF."""
    mistral_config = get_mistral_config()
    if not mistral_config:
        pytest.skip("Skipping Mistral OCR test: MISTRAL_API_KEY not set.")

    loader = DocumentLoaderMistralOCR(config=mistral_config)
    llm = LLM(get_lite_model())
    converter = MarkdownConverter(document_loader=loader, llm=llm)
    markdown_output = converter.to_markdown(PDF_PATH, vision=True)

    assert isinstance(markdown_output, str)
    assert len(markdown_output) > 0
    assert "Invoice" in markdown_output or "Bill To" in markdown_output

@pytest.mark.slow # Mark as slow due to external API call
def test_mistral_structured_conversion_image():
    """Test structured conversion with Mistral OCR on an image."""
    mistral_config = get_mistral_config()
    if not mistral_config:
        pytest.skip("Skipping Mistral OCR test: MISTRAL_API_KEY not set.")

    loader = DocumentLoaderMistralOCR(config=mistral_config)
    llm = LLM("gemini/gemini-2.0-flash")
    converter = MarkdownConverter(document_loader=loader, llm=llm)
    # converter.allow_verification = True
    
    terms = os.path.join(cwd, "tests", "files", "term_of_responsability.pdf")

    results = converter.to_markdown_structured(terms)
    
    # Convert PageContent objects to dictionaries before dumping to JSON
    content_list = [page.model_dump() if isinstance(page, PageContent) else page for page in results]
    content = json.dumps(content_list, indent=4)

    assert isinstance(results, list)
    assert len(results) == 1 # Expecting 1 page for an image
    assert isinstance(results[0], PageContent)
    assert len(results[0].items) > 0
    assert isinstance(results[0].items[0], ContentItem)

@pytest.mark.slow # Mark as slow due to external API call
def test_mistral_structured_conversion_pdf():
    """Test structured conversion with Mistral OCR on a PDF."""
    mistral_config = get_mistral_config()
    if not mistral_config:
        pytest.skip("Skipping Mistral OCR test: MISTRAL_API_KEY not set.")

    loader = DocumentLoaderMistralOCR(config=mistral_config)
    llm = LLM(get_lite_model())
    converter = MarkdownConverter(document_loader=loader, llm=llm)
    results = converter.to_markdown_structured(PDF_PATH, vision=True)

    assert isinstance(results, list)
    assert len(results) == 1 # Expecting 1 page for invoice.pdf
    assert isinstance(results[0], PageContent)
    assert len(results[0].items) > 0
    assert isinstance(results[0].items[0], ContentItem)

# === Error Handling Tests ===

def test_llm_required_error():
    """Test error when calling structured conversion without an LLM."""
    loader = DocumentLoaderPyPdf()
    converter = MarkdownConverter(document_loader=loader) # No LLM loaded
    with pytest.raises(ValueError, match="LLM is required for this operation but not set."):
        converter.to_markdown_structured(PDF_PATH, vision=False)

def test_no_document_loader_error():
    """Test error when calling conversion without a document loader."""
    llm = LLM(get_lite_model())
    converter = MarkdownConverter(llm=llm) # No loader loaded
    with pytest.raises(ValueError, match="Document loader is not set."):
        converter.to_markdown(PDF_PATH, vision=False)

    with pytest.raises(ValueError, match="Document loader is not set."):
        converter.to_markdown_structured(PDF_PATH, vision=False)

# === Async Method Tests (Basic Checks) ===

@pytest.mark.asyncio
async def test_pypdf_basic_async():
    loader = DocumentLoaderPyPdf()
    converter = MarkdownConverter(document_loader=loader)
    markdown_output = await converter.to_markdown_async(PDF_PATH, vision=False)
    assert isinstance(markdown_output, str)
    assert "Invoice" in markdown_output

@pytest.mark.asyncio
@pytest.mark.slow # Mark as slow because it uses an LLM API call
async def test_pypdf_structured_async():
    loader = DocumentLoaderPyPdf()
    llm = LLM(get_lite_model())
    converter = MarkdownConverter(document_loader=loader, llm=llm)
    results = await converter.to_markdown_structured_async(PDF_PATH, vision=True)
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], PageContent)


if __name__ == "__main__":
    # from litellm import completion
    # import os
    
    # response = completion(
    #     model="gemini/gemini-2.0-flash", 
    #     messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}]
    # )

    # print(response.choices[0].message.content)

    test_mistral_structured_conversion_image()
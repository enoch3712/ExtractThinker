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
    """Test basic Markdown conversion with PyPDF, vision off.
    This now requires an LLM as per the updated implementation."""
    loader = DocumentLoaderPyPdf()
    llm = LLM(get_lite_model())  # LLM is now required
    converter = MarkdownConverter(document_loader=loader, llm=llm)
    markdown_output = converter.to_markdown(PDF_PATH, vision=False)

    assert isinstance(markdown_output, list)
    assert len(markdown_output) > 0
    assert all(isinstance(item, str) for item in markdown_output)

    assert "0012" in markdown_output[0]

def test_pypdf_basic_conversion_with_vision():
    """Test basic Markdown conversion with PyPDF, vision on.
    This now requires an LLM as per the updated implementation."""
    loader = DocumentLoaderPyPdf()
    llm = LLM(get_lite_model())  # LLM is now required
    converter = MarkdownConverter(document_loader=loader, llm=llm)
    markdown_output = converter.to_markdown(PDF_PATH, vision=True)

    assert isinstance(markdown_output, list)
    assert len(markdown_output) > 0
    assert all(isinstance(item, str) for item in markdown_output)

    assert "0012" in markdown_output[0]

def test_pypdf_llm_conversion_with_vision():
    """Test LLM-based Markdown conversion with PyPDF, vision on."""
    loader = DocumentLoaderPyPdf()
    llm = LLM(get_lite_model())
    converter = MarkdownConverter(document_loader=loader, llm=llm)
    markdown_output = converter.to_markdown(PDF_PATH, vision=True)

    assert isinstance(markdown_output, list)
    assert len(markdown_output) > 0
    assert all(isinstance(item, str) for item in markdown_output)

    assert "0012" in markdown_output[0]

def test_pypdf_structured_conversion():
    """Test structured conversion with PyPDF, vision off."""
    loader = DocumentLoaderPyPdf()
    llm = LLM(get_lite_model())
    converter = MarkdownConverter(document_loader=loader, llm=llm)
    results = converter.to_markdown_structured(PDF_PATH)

    assert isinstance(results, list)
    assert all(isinstance(item, PageContent) for item in results)
    assert len(results) == 1
    
    # Check for "0012" in any content item
    page_content = results[0]
    found_0012 = False
    for item in page_content.items:
        if "0012" in item.content and item.certainty > 9:
            found_0012 = True
            break
    assert found_0012, "Could not find '0012' in any content item"

def test_spreadsheet_basic_conversion():
    """Test basic Markdown conversion with SpreadSheet.
    This now requires an LLM as per the updated implementation."""
    try:
        loader = DocumentLoaderSpreadSheet()
    except ImportError as e:
        pytest.skip(f"Skipping spreadsheet test: {e}")

    llm = LLM(get_lite_model())  # LLM is now required
    converter = MarkdownConverter(document_loader=loader, llm=llm)
    markdown_output = converter.to_markdown(SPREADSHEET_PATH, vision=False)

    assert isinstance(markdown_output, list)
    assert len(markdown_output) > 0
    assert all(isinstance(item, str) for item in markdown_output)
    
    # Check for content expected from family_budget.xlsx
    assert any("Monthly Income" in page for page in markdown_output)

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

    assert isinstance(markdown_output, list)
    assert len(markdown_output) > 0
    assert all(isinstance(item, str) for item in markdown_output)
    # Check for content expected from the image OCR in at least one page
    assert any("Invoice" in page or "Example Corp" in page for page in markdown_output)

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

    assert isinstance(markdown_output, list)
    assert len(markdown_output) == 1
    assert isinstance(markdown_output[0], str)
    assert "0012" in markdown_output[0]

@pytest.mark.slow # Mark as slow due to external API call
def test_mistral_structured_conversion_image():
    """Test structured conversion with Mistral OCR on an image."""
    mistral_config = get_mistral_config()
    if not mistral_config:
        pytest.skip("Skipping Mistral OCR test: MISTRAL_API_KEY not set.")

    loader = DocumentLoaderMistralOCR(config=mistral_config)
    llm = LLM(get_lite_model())
    converter = MarkdownConverter(document_loader=loader, llm=llm)
    # converter.allow_verification = True
    
    terms = os.path.join(cwd, "tests", "test_images", "invoice.png")

    results = converter.to_markdown_structured(terms)

    assert isinstance(results, list)
    assert len(results) == 1
    # Results is now a list of strings, not PageContent objects
    assert isinstance(results[0], str)
    assert "0000001" in results[0]

@pytest.mark.slow # Mark as slow due to external API call
def test_mistral_structured_conversion_pdf():
    """Test structured conversion with Mistral OCR on a PDF."""
    mistral_config = get_mistral_config()
    if not mistral_config:
        pytest.skip("Skipping Mistral OCR test: MISTRAL_API_KEY not set.")

    loader = DocumentLoaderMistralOCR(config=mistral_config)
    llm = LLM(get_lite_model())
    converter = MarkdownConverter(document_loader=loader, llm=llm)
    # converter.allow_verification = True
    
    terms = os.path.join(cwd, "tests", "files", "invoice.pdf")

    results = converter.to_markdown_structured(terms)

    assert isinstance(results, list)
    assert len(results) == 1
    # Results is now a list of strings, not PageContent objects
    assert isinstance(results[0], str)
    assert "0012" in results[0]

# === Error Handling Tests ===

def test_llm_required_error():
    """Test error when calling conversion methods without an LLM."""
    loader = DocumentLoaderPyPdf()
    converter = MarkdownConverter(document_loader=loader)  # No LLM loaded
    
    # to_markdown_structured should fail without LLM
    with pytest.raises(ValueError, match="LLM is required for this operation but not set."):
        converter.to_markdown_structured(PDF_PATH, vision=False)
    
    # to_markdown should now also fail without LLM
    with pytest.raises(ValueError, match="LLM is required for markdown conversion but not configured."):
        converter.to_markdown(PDF_PATH, vision=False)

def test_no_document_loader_error():
    """Test behavior when calling methods without a document loader."""
    llm = LLM(get_lite_model())
    converter = MarkdownConverter(llm=llm)  # No loader loaded
    
    # to_markdown_structured should still fail without document loader
    with pytest.raises(ValueError, match="Document loader is not set."):
        converter.to_markdown_structured(PDF_PATH, vision=False)
    
    # to_markdown should now work with just an LLM for simple text files
    test_content = "# Test Heading\nThis is some test content."
    test_file = "test_content.txt"
    with open(test_file, "w") as f:
        f.write(test_content)
    
    try:
        markdown_output = converter.to_markdown(test_file, vision=False)
        assert isinstance(markdown_output, list)
        assert len(markdown_output) == 1
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

# === Async Method Tests (Basic Checks) ===

@pytest.mark.asyncio
async def test_pypdf_basic_async():
    loader = DocumentLoaderPyPdf()
    llm = LLM(get_lite_model())  # LLM is now required
    converter = MarkdownConverter(document_loader=loader, llm=llm)
    markdown_output = await converter.to_markdown_async(PDF_PATH, vision=False)
    assert isinstance(markdown_output, list)
    assert len(markdown_output) > 0
    assert all(isinstance(page, str) for page in markdown_output)
    assert any("Invoice" in page for page in markdown_output)

@pytest.mark.asyncio
@pytest.mark.slow # Mark as slow because it uses an LLM API call
async def test_pypdf_structured_async():
    loader = DocumentLoaderPyPdf()
    llm = LLM(get_lite_model())
    converter = MarkdownConverter(document_loader=loader, llm=llm)
    results = await converter.to_markdown_structured_async(PDF_PATH, vision=True)
    assert isinstance(results, list)
    assert len(results) == 1
    # Results is now a list of strings, not PageContent objects
    assert isinstance(results[0], str)

# Add a test for LLM only (no document loader)
@pytest.mark.slow
def test_llm_only_markdown_conversion():
    """Test markdown conversion with just an LLM (no document loader)."""
    llm = LLM(get_lite_model())
    converter = MarkdownConverter(llm=llm)  # No document_loader
    
    # Create a simple text file to test
    test_content = "# Test Heading\nThis is some test content."
    test_file = "test_content.txt"
    with open(test_file, "w") as f:
        f.write(test_content)
    
    try:
        # Should work with just an LLM
        markdown_output = converter.to_markdown(test_file, vision=False)
        assert isinstance(markdown_output, list)
        assert len(markdown_output) == 1  # Should be a single element list
        assert isinstance(markdown_output[0], str)
        assert "Heading" in markdown_output[0]
    finally:
        # Clean up the test file
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    # from litellm import completion
    # import os
    
    # response = completion(
    #     model="gemini/gemini-2.0-flash", 
    #     messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}]
    # )

    # print(response.choices[0].message.content)

    test_mistral_structured_conversion_image()
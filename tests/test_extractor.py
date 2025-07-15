import asyncio
import os
import sys

from extract_thinker.document_loader.document_loader_aws_textract import DocumentLoaderAWSTextract
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
from extract_thinker.extractor import Extractor
from extract_thinker.document_loader.document_loader_tesseract import DocumentLoaderTesseract
from extract_thinker.document_loader.document_loader_pypdf import DocumentLoaderPyPdf
from extract_thinker.document_loader.document_loader_spreadsheet import DocumentLoaderSpreadSheet
from extract_thinker.llm import LLM, LLMEngine
from extract_thinker.models.completion_strategy import CompletionStrategy
from tests.models.invoice import InvoiceContract
from tests.models.ChartWithContent import ChartWithContent
from tests.models.page_contract import ReportContract
from tests.models.gdp_contract import EUData, EUDataOptional
from extract_thinker.document_loader.document_loader_azure_document_intelligence import DocumentLoaderAzureForm
import pytest
import numpy as np
from litellm import embedding
from extract_thinker.document_loader.document_loader_docling import DocumentLoaderDocling
from tests.models.handbook_contract import HandbookContract
from extract_thinker.global_models import get_gpt_o4_model, get_lite_model, get_big_model, get_gemini_flash_model, get_gpt_mini_model
from pydantic import BaseModel, Field
from extract_thinker.exceptions import ExtractThinkerError
from typing import List, Optional


load_dotenv()
cwd = os.getcwd()

# Define a new Pydantic model for budget data
class BudgetData(BaseModel):
    """Model for budget data extraction from spreadsheet"""
    total_income_projected: float
    total_expense_projected: float
    cash_flow_projected: float
    monthly_income_items: Optional[List[str]] = None
    monthly_expense_items: Optional[List[str]] = None


def test_extract_with_pypdf_and_gpt4o_mini_vision():

    # Arrange
    test_file_path = os.path.join(cwd, "tests", "test_images", "invoice.png")

    extractor = Extractor()
    extractor.load_document_loader(
        DocumentLoaderPyPdf()
    )
    extractor.load_llm(get_gpt_mini_model())

    # Act
    result = extractor.extract(test_file_path, InvoiceContract, vision=True)

    # Assert
    assert result is not None
    assert result.invoice_number == "0000001"
    assert result.invoice_date == "2014-05-07"

def test_vision_content_pdf():
    # Arrange
    extractor = Extractor()
    extractor.load_llm(get_lite_model())
    test_file_path = os.path.join(cwd, "tests", "files", "invoice.pdf")

    # Act
    result = extractor.extract(test_file_path, InvoiceContract, vision=True)

    # Assert
    assert result is not None
    
    # Check invoice details
    assert result.invoice_number == "00012"
    assert result.invoice_date == "1/30/23"
    assert result.total_amount == 1125

    # Check line items
    assert len(result.lines) == 1
    line = result.lines[0]
    assert line.description == "Consultation services"
    assert line.quantity == 3  # 3.0 hours
    assert line.unit_price == 375  # Rate per hour
    assert line.amount == 1125  # Total amount for the line

def test_chart_with_content():
    # Arrange
    extractor = Extractor()
    extractor.load_llm(get_lite_model())
    test_file_path = os.path.join(cwd, "tests", "test_images", "eu_tax_chart.png")

    # Act
    result = extractor.extract(test_file_path, ChartWithContent, vision=True)

    # Assert
    assert result is not None
    
    # Test content
    assert "In 2022, total tax revenues grew below nominal GDP in 15 Member States" in result.content
    assert "tax revenues (numerator) did not grow as fast as nominal GDP (denominator)" in result.content
    
    # Test chart properties
    assert result.chart is not None

def test_extract_with_loader_and_vision():
    # Arrange
    test_file_path = os.path.join(cwd, "tests", "files", "invoice.pdf")

    extractor = Extractor()
    loader = DocumentLoaderPyPdf()
    extractor.load_document_loader(loader)
    extractor.load_llm(get_lite_model())

    # Act
    result = extractor.extract(test_file_path, InvoiceContract, vision=True)

    # Assert
    assert result.invoice_number == "00012"
    assert result.invoice_date == "1/30/23"
    assert result.total_amount == 1125

    # Check line items
    assert len(result.lines) == 1
    line = result.lines[0]
    assert line.description == "Consultation services"
    assert line.quantity == 3  # 3.0 hours
    assert line.unit_price == 375  # Rate per hour
    assert line.amount == 1125  # Total amount for the line

def test_extract_with_invalid_file_path():
    # Arrange
    extractor = Extractor()
    extractor.load_llm(get_lite_model())
    invalid_file_path = os.path.join(cwd, "tests", "nonexistent", "fake_file.png")

    # Act & Assert
    with pytest.raises(ExtractThinkerError) as exc_info:
        extractor.extract(invalid_file_path, InvoiceContract, vision=True)
    
    assert "Failed to extract from source: Cannot handle source" in str(exc_info.value)

def test_forbidden_strategy_with_token_limit():
    test_file_path = os.path.join(os.getcwd(), "tests", "test_images", "eu_tax_chart.png")
    tesseract_path = os.getenv("TESSERACT_PATH")

    llm = LLM(get_lite_model(), token_limit=10)

    extractor = Extractor()
    extractor.load_document_loader(DocumentLoaderTesseract(tesseract_path))
    extractor.load_llm(llm)

    # Should raise ExtractThinkerError due to FORBIDDEN strategy
    with pytest.raises(ExtractThinkerError, match="Incomplete output received and FORBIDDEN strategy is set"):
        extractor.extract(
            test_file_path,
            ReportContract,
            vision=False,
            content="RULE: Give me all the pages content",
            completion_strategy=CompletionStrategy.FORBIDDEN
        )

async def extract_async(extractor, file_path, vision, completion_strategy):
    return extractor.extract(
        file_path,
        EUData,
        vision=vision,
        completion_strategy=completion_strategy
    )

def test_pagination_handler():
    test_file_path = os.path.join(os.getcwd(), "tests", "files", "Regional_GDP_per_capita_2018_2.pdf")

    extractor = Extractor()
    extractor.load_document_loader(DocumentLoaderDocling())
    extractor.load_llm(get_big_model())

    # Create and run both extractions in parallel
    async def run_parallel_extractions():
        result_1, result_2 = await asyncio.gather(
            extract_async(extractor, test_file_path, vision=True, completion_strategy=CompletionStrategy.PAGINATE),
            extract_async(extractor, test_file_path, vision=True, completion_strategy=CompletionStrategy.FORBIDDEN)
        )
        return result_1, result_2

    # Run the async extraction and get the results as instances of OptionalEUData
    results = asyncio.run(run_parallel_extractions())
    result_1, result_2 = results

    # Compare top-level EU data
    assert result_1.eu_total_gdp_million_27 == result_2.eu_total_gdp_million_27
    assert result_1.eu_total_gdp_million_28 == result_2.eu_total_gdp_million_28

    # Compare country count
    assert len(result_1.countries) == len(result_2.countries)

    # Compare regions count for each country
    # for country1 in result_1.countries:
    #     matching_country = next(
    #         (c for c in result_2.countries if c.country == country1.country), 
    #         None
    #     )
    #     assert matching_country is not None, f"Country {country1.country} not found in result_2"
    #     assert len(country1.regions) == len(matching_country.regions)

    # Keeping detailed comparison code commented for future reference
    #     # Compare region-level data
    #     for region1 in country1.regions:
    #         matching_region = next(
    #             (r for r in matching_country.regions if r.region == region1.region),
    #             None
    #         )
    #         assert matching_region is not None, f"Region {region1.region} not found in country {country1.country}"
    #         
    #         # Compare region data
    #         assert region1.gdp_million == matching_region.gdp_million
    #         assert region1.share_in_eu27_gdp == matching_region.share_in_eu27_gdp
    #         assert region1.gdp_per_capita == matching_region.gdp_per_capita
    #         
    #         # Compare province-level data
    #         assert len(region1.provinces) == len(matching_region.provinces)
    #         for province1 in region1.provinces:
    #             matching_province = next(
    #                 (p for p in matching_region.provinces if p.name == province1.name),
    #                 None
    #             )
    #             assert matching_province is not None, f"Province {province1.name} not found in region {region1.region}"
    #             
    #             # Compare province data
    #             assert province1.gdp_million == matching_province.gdp_million
    #             assert province1.share_in_eu27_gdp == matching_province.share_in_eu27_gdp
    #             assert province1.gdp_per_capita == matching_province.gdp_per_capita

def test_pagination_handler_optional():
    test_file_path = os.path.join(os.getcwd(), "tests", "files", "Regional_GDP_per_capita_2018_2.pdf")

    extractor = Extractor()
    extractor.load_document_loader(DocumentLoaderDocling())
    extractor.load_llm(get_big_model())

    async def extract_async_optional(extractor, file_path, vision, completion_strategy):
        return extractor.extract(
            file_path,
            EUDataOptional,
            vision=vision,
            completion_strategy=completion_strategy
        )
    
    result = asyncio.run(extract_async_optional(extractor, test_file_path, vision=True, completion_strategy=CompletionStrategy.PAGINATE))

    assert len(result.countries) == 6

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    response = embedding(
        model=model,  # OpenAI's embedding model
        input=[text]
    )
    return response.data[0]['embedding']

def cosine_similarity(v1, v2):
    # Convert to numpy arrays if they aren't already
    v1, v2 = np.array(v1), np.array(v2)
    # Calculate cosine similarity
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def semantically_similar(text1, text2, threshold=0.9):
    # Normalize texts
    text1 = text1.lower().strip()
    text2 = text2.lower().strip()
    
    # Get embeddings
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)
    
    # Calculate similarity
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity >= threshold

def test_concatenation_handler():
    test_file_path = os.path.join(os.getcwd(), "tests", "test_images", "eu_tax_chart.png")
    tesseract_path = os.getenv("TESSERACT_PATH")
    extractor = Extractor()
    extractor.load_document_loader(DocumentLoaderTesseract(tesseract_path))
    llm_first = LLM(get_big_model(), token_limit=4096)
    extractor.load_llm(llm_first)

    result_1: ReportContract = extractor.extract(
        test_file_path,
        ReportContract,
        vision=True,
        completion_strategy=CompletionStrategy.CONCATENATE
    )

    second_extractor = Extractor()
    second_extractor.load_document_loader(DocumentLoaderTesseract(tesseract_path))
    second_extractor.load_llm(get_big_model())

    result_2: ReportContract = second_extractor.extract(
        test_file_path,
        ReportContract,
        vision=True,
        completion_strategy=CompletionStrategy.FORBIDDEN
    )

    assert semantically_similar(
        result_1.title,
        result_2.title,
        threshold=0.8
    ), "Titles are not semantically similar enough (threshold: 90%)"

    assert result_1.pages[0].number == result_2.pages[0].number
    assert semantically_similar(
        result_1.pages[0].content, 
        result_2.pages[0].content,
        threshold=0.8
    ), "Page contents are not semantically similar enough (threshold: 90%)"

def test_llm_timeout():
    # Arrange
    test_file_path = os.path.join(cwd, "tests", "files", "invoice.pdf")
    
    extractor = Extractor()
    extractor.load_document_loader(DocumentLoaderPyPdf())
    
    # Create LLM with very short timeout
    llm = LLM(get_lite_model())
    llm.set_timeout(1)  # Set timeout to 1ms (extremely short to force timeout)
    extractor.load_llm(llm)
    
    # Act & Assert
    with pytest.raises(Exception) as exc_info:
        extractor.extract(test_file_path, InvoiceContract)
    
    # Reset timeout to normal value
    llm.set_timeout(3000)
    
    # Verify normal operation works after reset
    result = extractor.extract(test_file_path, InvoiceContract)
    assert result is not None

def test_extract_with_default_backend():
    """Test extraction using default LiteLLM backend"""
    # Arrange
    test_file_path = os.path.join(cwd, "tests", "files", "invoice.pdf")
    
    extractor = Extractor()
    extractor.load_document_loader(DocumentLoaderPyPdf())
    extractor.load_llm(LLM(get_lite_model(), backend=LLMEngine.DEFAULT))

    # Act
    result = extractor.extract(test_file_path, InvoiceContract)

    # Assert
    assert result is not None
    assert result.invoice_number == "00012"
    assert result.invoice_date == "1/30/23"
    assert result.total_amount == 1125

def test_extract_with_pydanticai_backend():
    """Test extraction using PydanticAI backend if available"""
    try:
        # Arrange
        test_file_path = os.path.join(cwd, "tests", "files", "invoice.pdf")
        
        extractor = Extractor()
        extractor.load_document_loader(DocumentLoaderPyPdf())
        extractor.load_llm(LLM(get_gpt_o4_model(), backend=LLMEngine.PYDANTIC_AI))

        # Act
        result = extractor.extract(test_file_path, InvoiceContract)

        # Assert
        assert result is not None
        assert result.invoice_number == "00012"
        assert result.invoice_date == "1/30/23"
        assert result.total_amount == 1125

    except ImportError:
        pytest.skip("pydantic-ai not installed")

def test_extract_from_url_docling_and_gpt4o_mini():
    """
    Test extraction from a URL using the Docling document loader and gpt-4o-mini LLM.
    The test asserts that the extracted title is as expected.
    """
    url = "https://ai.pydantic.dev/models/openai/"

    # Initialize the extractor, load the Docling loader and the gpt-4o-mini LLM
    extractor = Extractor()
    extractor.load_document_loader(DocumentLoaderDocling())
    extractor.load_llm(get_lite_model())

    # Act: Extract the document using the specified URL and the HandbookContract
    result: HandbookContract = extractor.extract(url, HandbookContract)

    #TODO handbook is 403 error now. So we need to refactor the test to use a different url.

    # Check handbook data
    #assert "FCA Handbook" in result.title, f"Expected title to contain 'FCA Handbook', but got: {result.title}"
    assert result.title == "OpenAI"

def test_spreadsheet_data_extraction():
    """
    Test that the Excel spreadsheet data (not just sheet names) is properly passed to LLM.
    This test verifies the fix for the bug where only sheet names were passed to LLM.
    """
    # Arrange
    budget_file_path = os.path.join(cwd, "tests", "files", "family_budget.xlsx")
    
    # Skip if file doesn't exist
    if not os.path.exists(budget_file_path):
        pytest.skip(f"Test file {budget_file_path} not found")
    
    # Create extractor with spreadsheet loader
    extractor = Extractor()
    extractor.load_document_loader(DocumentLoaderSpreadSheet())
    extractor.load_llm(get_lite_model())
    
    # Act
    result = extractor.extract(budget_file_path, BudgetData)
    
    # Assert
    assert result is not None
    assert isinstance(result, BudgetData)
    
    # Check budget values from the spreadsheet
    assert result.total_income_projected == 5700, "Total income should be extracted from spreadsheet data"
    assert result.total_expense_projected == 3603, "Total expense should be extracted from spreadsheet data"
    assert result.cash_flow_projected == 2097, "Cash flow should be extracted from spreadsheet data"
    
    # If expense items were extracted, verify some common categories
    if result.monthly_expense_items:
        expense_categories_text = " ".join(result.monthly_expense_items).lower()
        expected_categories = ["housing", "groceries", "utilities", "transportation"]
        found_categories = [cat for cat in expected_categories if cat.lower() in expense_categories_text]
        assert len(found_categories) > 0, "Should find at least one expense category"

def test_extract_from_multiple_sources():
    """
    Test extracting from multiple sources (PDF and URL) in a single call.
    Combines invoice data with handbook data using DocumentLoaderDocling.
    """
    # Arrange
    pdf_path = os.path.join(cwd, "tests", "files", "invoice.pdf")
    url = "https://ai.pydantic.dev/models/openai/"

    extractor = Extractor()
    docling_loader = DocumentLoaderDocling()
    extractor.load_document_loader(docling_loader)
    extractor.load_llm(get_big_model())

    class CombinedData(BaseModel):
        invoice_number: str
        invoice_date: str
        total_amount: float
        handbook_title: str = Field(alias="title of the url, and not the invoice")

    # Act
    result: CombinedData = extractor.extract(
        [pdf_path, url],
        CombinedData,
    )

    # Assert
    # Check invoice data
    assert result.invoice_number == "00012"
    assert result.invoice_date == "1/30/23"
    assert result.total_amount == 1125

    # Check handbook data
    #assert "FCA Handbook" in result.handbook_title, f"Expected title to contain 'FCA Handbook', but got: {result.handbook_title}"
    assert result.handbook_title == "OpenAI"

def test_thinking_mode_gemini_flash():
    """Test thinking mode with Gemini Flash model."""
    class User(BaseModel):
        name: str
        age: int

    llm = LLM(get_gemini_flash_model())
    extractor = Extractor(llm=llm)
    extractor.enable_thinking_mode(True)
    extractor.set_page_count(1)

    result = extractor.extract(
        source={'content': "My name is John and I am 30 years old."},
        response_model=User
    )
    assert result.name == "John"
    assert result.age == 30

def test_thinking_mode_gpt_mini():
    """Test thinking mode with GPT-4o Mini model."""
    class User(BaseModel):
        name: str
        age: int

    llm = LLM(get_gpt_mini_model())
    extractor = Extractor(llm=llm)
    extractor.enable_thinking_mode(True)
    extractor.set_page_count(1)

    result = extractor.extract(
        source={'content': "My name is Jane and I am 25 years old."},
        response_model=User
    )
    assert result.name == "Jane"
    assert result.age == 25

if __name__ == "__main__":
    test_chart_with_content()
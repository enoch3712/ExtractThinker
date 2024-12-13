import asyncio
import os
from typing import List, Optional
from pydantic import Field
import time
from dotenv import load_dotenv
from extract_thinker.document_loader.document_loader_aws_textract import DocumentLoaderAWSTextract
from extract_thinker.document_loader.document_loader_pdfplumber import DocumentLoaderPdfPlumber
from extract_thinker.extractor import Extractor
from extract_thinker.document_loader.document_loader_tesseract import DocumentLoaderTesseract
from extract_thinker.document_loader.document_loader_pypdf import DocumentLoaderPyPdf
from extract_thinker.llm import LLM
from extract_thinker.models.completion_strategy import CompletionStrategy
from extract_thinker.models.contract import Contract
from tests.models.invoice import InvoiceContract
from tests.models.ChartWithContent import ChartWithContent
from extract_thinker.document_loader.document_loader_azure_document_intelligence import DocumentLoaderAzureForm
import pytest

load_dotenv()
cwd = os.getcwd()

def test_extract_with_tessaract_and_gpt4o_mini():

    # Arrange
    tesseract_path = os.getenv("TESSERACT_PATH")
    test_file_path = os.path.join(cwd, "tests", "test_images", "invoice.png")

    extractor = Extractor()
    extractor.load_document_loader(
        DocumentLoaderTesseract(tesseract_path)
    )
    extractor.load_llm("gpt-4o-mini")

    # Act
    result = extractor.extract(test_file_path, InvoiceContract)

    # Assert
    assert result is not None
    assert result.invoice_number == "0000001"
    assert result.invoice_date == "2014-05-07"

def test_extract_with_azure_di_and_gpt4o_mini():
    subscription_key = os.getenv("AZURE_SUBSCRIPTION_KEY")
    endpoint = os.getenv("AZURE_ENDPOINT")
    test_file_path = os.path.join(cwd, "tests", "test_images", "invoice.png")

    extractor = Extractor()
    extractor.load_document_loader(
        DocumentLoaderAzureForm(subscription_key, endpoint)
    )
    extractor.load_llm("gpt-4o-mini")
    # Act
    result = extractor.extract(test_file_path, InvoiceContract)

    # Assert
    assert result is not None
    assert result.lines[0].description == "Website Redesign"
    assert result.lines[0].quantity == 1
    assert result.lines[0].unit_price == 2500
    assert result.lines[0].amount == 2500

def test_extract_with_pypdf_and_gpt4o_mini():
    test_file_path = os.path.join(cwd, "tests", "files", "invoice.pdf")

    extractor = Extractor()
    document_loader = DocumentLoaderPyPdf()
    extractor.load_document_loader(document_loader)
    extractor.load_llm("gpt-4o-mini")
    
    # Act
    result = extractor.extract(test_file_path, InvoiceContract)

    # Assert
    assert result is not None
    assert result.lines[0].description == "Consultation services"
    assert result.lines[0].quantity == 3
    assert result.lines[0].unit_price == 375
    assert result.lines[0].amount == 1125

def test_vision_content_pdf():
    # Arrange
    extractor = Extractor()
    extractor.load_llm("gpt-4o-mini")
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
    extractor.load_llm("gpt-4o-mini")
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
    extractor.load_llm("gpt-4o-mini")

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
    extractor.load_llm("gpt-4o-mini")
    invalid_file_path = os.path.join(cwd, "tests", "nonexistent", "fake_file.png")

    # Act & Assert
    with pytest.raises(ValueError) as exc_info:
        extractor.extract(invalid_file_path, InvoiceContract, vision=True)
    
    assert "does not exist" in str(exc_info.value)

def test_batch_extraction_single_source():
    # Arrange
    load_dotenv()
    tesseract_path = os.getenv("TESSERACT_PATH")
    test_file_path = os.path.join(os.getcwd(), "tests", "test_images", "invoice.png")

    extractor = Extractor()
    extractor.load_document_loader(DocumentLoaderTesseract(tesseract_path))
    extractor.load_llm("gpt-4o-mini")

    # Act
    batch_job = extractor.extract_batch(test_file_path, InvoiceContract)
    
    # Assert batch status
    status = asyncio.run(batch_job.get_status())
    assert status in ["queued", "processing", "completed"]
    print(f"Batch status: {status}")

    result = asyncio.run(batch_job.get_result())

    # Get results and verify
    assert result.invoice_number == "0000001"
    assert result.invoice_date == "2014-05-07"

def test_cancel_batch_extraction():
    # Arrange
    tesseract_path = os.getenv("TESSERACT_PATH") 
    test_file_path = os.path.join(os.getcwd(), "tests", "test_images", "invoice.png")
    batch_file_path = os.path.join(os.getcwd(), "tests", "batch_input.jsonl")
    output_file_path = os.path.join(os.getcwd(), "tests", "batch_output.jsonl")

    extractor = Extractor()
    extractor.load_document_loader(DocumentLoaderTesseract(tesseract_path))
    extractor.load_llm("gpt-4o-mini")

    # Act
    batch_job = extractor.extract_batch(
        test_file_path, 
        InvoiceContract,
        batch_file_path=batch_file_path,
        output_file_path=output_file_path
    )
    
    # Cancel the batch job
    cancel_success = asyncio.run(batch_job.cancel())
    assert cancel_success, "Batch job cancellation failed"

    # Add a small delay to ensure cleanup has time to complete
    time.sleep(1)

    # Check if files were removed
    assert not os.path.exists(batch_job.file_path), f"Batch input file was not removed: {batch_job.file_path}"
    assert not os.path.exists(batch_job.output_path), f"Batch output file was not removed: {batch_job.output_path}"

class PageContract(Contract):
    title: str
    number: int
    content: str = Field(description="Give me all the content, word for word")

class ReportContract(Contract):
    title: str
    pages: List[PageContract]

class PaginateContract(Contract):
    us_state: str
    driver_license_number: str
    expiration_date: str
    registered_for_personal_use: bool

class CountryGDP(Contract):
    country: str
    location: str
    gdp_real: float

class ProvinceData(Contract):
    name: str
    gdp_million: Optional[float] = Field(None, description="GDP (€ million)")
    share_in_eu27_gdp: Optional[float] = Field(None, description="Share in EU27/national GDP (%)")
    gdp_per_capita: Optional[int] = Field(None, description="GDP per capita (€)")

class RegionData(Contract):
    region: str
    gdp_million: Optional[float] = Field(None, description="GDP (€ million)")
    share_in_eu27_gdp: Optional[float] = Field(None, description="Share in EU27/national GDP (%)")
    gdp_per_capita: Optional[int] = Field(None, description="GDP per capita (€)")
    provinces: List[ProvinceData] = Field(default_factory=list)

class CountryData(Contract):
    country: str
    total_gdp_million: Optional[float] = Field(None, description="Total GDP (€ million)")
    regions: List[RegionData]

class EUData(Contract):
    eu_total_gdp_million_27: float = Field(None, description="EU27 Total GDP (€ million)")
    eu_total_gdp_million_28: float = Field(None, description="EU28 Total GDP (€ million)")
    countries: List[CountryData]
    
def test_data_long_text():
    test_file_path = os.path.join(os.getcwd(), "tests", "test_images", "eu_tax_chart.png")
    tesseract_path = os.getenv("TESSERACT_PATH")

    extractor = Extractor()
    extractor.load_document_loader(DocumentLoaderTesseract(tesseract_path))
    extractor.load_llm("gpt-4o-mini")

    result = extractor.extract(
        test_file_path,
        ReportContract,
        vision=True,
        content="RULE: Give me all the pages content",
        completion_strategy=CompletionStrategy.FORBIDDEN
    )
    pass

def test_forbidden_strategy_with_token_limit():
    test_file_path = os.path.join(os.getcwd(), "tests", "test_images", "eu_tax_chart.png")
    tesseract_path = os.getenv("TESSERACT_PATH")

    llm = LLM("gpt-4o-mini", token_limit=10)

    extractor = Extractor()
    extractor.load_document_loader(DocumentLoaderTesseract(tesseract_path))
    extractor.load_llm(llm)

    # Should raise ValueError due to FORBIDDEN strategy
    with pytest.raises(ValueError, match="Incomplete output received and FORBIDDEN strategy is set"):
        extractor.extract(
            test_file_path,
            ReportContract,
            vision=False,
            content="RULE: Give me all the pages content",
            completion_strategy=CompletionStrategy.FORBIDDEN
        )

def test_pagination_handler():
    test_file_path = os.path.join(os.getcwd(), "tests", "test_images", "eu_tax_chart.png")
    tesseract_path = os.getenv("TESSERACT_PATH")

    extractor = Extractor()
    extractor.load_document_loader(DocumentLoaderTesseract(tesseract_path))
    extractor.load_llm("gpt-4o")

    result_1: ReportContract = extractor.extract(
        test_file_path,
        ReportContract,
        vision=False,
        completion_strategy=CompletionStrategy.CONCATENATE
    )

    result_1_json = result_1.model_dump_json()

    result_2: EUData = extractor.extract(
        test_file_path,
        EUData,
        vision=True, 
        completion_strategy=CompletionStrategy.FORBIDDEN
    )

    # Compare top-level EU data
    assert result_1.eu_total_gdp_million_27 == result_2.eu_total_gdp_million_27
    assert result_1.eu_total_gdp_million_28 == result_2.eu_total_gdp_million_28

    # Compare country-level data
    for country1 in result_1.countries:
        # Find matching country in result_2
        matching_country = next(
            (c for c in result_2.countries if c.country == country1.country), 
            None
        )
        assert matching_country is not None, f"Country {country1.country} not found in result_2"
        
        # Compare country data
        assert country1.total_gdp_million == matching_country.total_gdp_million
        assert len(country1.regions) == len(matching_country.regions)
        
        # Compare region-level data
        for region1 in country1.regions:
            matching_region = next(
                (r for r in matching_country.regions if r.region == region1.region),
                None
            )
            assert matching_region is not None, f"Region {region1.region} not found in country {country1.country}"
            
            # Compare region data
            assert region1.gdp_million == matching_region.gdp_million
            assert region1.share_in_eu27_gdp == matching_region.share_in_eu27_gdp
            assert region1.gdp_per_capita == matching_region.gdp_per_capita
            
            # Compare province-level data
            assert len(region1.provinces) == len(matching_region.provinces)
            for province1 in region1.provinces:
                matching_province = next(
                    (p for p in matching_region.provinces if p.name == province1.name),
                    None
                )
                assert matching_province is not None, f"Province {province1.name} not found in region {region1.region}"
                
                # Compare province data
                assert province1.gdp_million == matching_province.gdp_million
                assert province1.share_in_eu27_gdp == matching_province.share_in_eu27_gdp
                assert province1.gdp_per_capita == matching_province.gdp_per_capita

if __name__ == "__main__":
    # test_data_long_text()
    # test_forbidden_strategy_with_token_limit()
    test_pagination_handler()
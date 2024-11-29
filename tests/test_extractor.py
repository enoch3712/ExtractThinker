import asyncio
import os
import time
from dotenv import load_dotenv
from extract_thinker.extractor import Extractor
from extract_thinker.document_loader.document_loader_tesseract import DocumentLoaderTesseract
from extract_thinker.document_loader.document_loader_pypdf import DocumentLoaderPyPdf
from tests.models.invoice import InvoiceContract
from tests.models.ChartWithContent import ChartWithContent
from extract_thinker.document_loader.document_loader_azure_document_intelligence import DocumentLoaderAzureForm

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
    test_file_path = os.path.join(cwd, "tests", "files", "eu_tax_chart.png")

    # Act
    result = extractor.extract(test_file_path, ChartWithContent, vision=True)

    # Assert
    assert result is not None
    
    # Test content
    assert "In 2022, total tax revenues grew below nominal GDP in 15 Member States" in result.content
    assert "tax revenues (numerator) did not grow as fast as nominal GDP (denominator)" in result.content
    
    # Test chart properties
    assert result.chart is not None
    assert result.chart.title == "Figure 3: Tax revenues in the EU since 2009 (nominal terms and percentage of GDP)"
    assert result.chart.source == "Eurostat [gov_10a_taxag], as of 31 January 2024. Nominal values converted in EUR for non-EA countries."
    assert result.chart.type == "combination"  # Bar and line chart
    assert len(result.chart.data_series) == 2
    assert "Nominal value" in [series.name for series in result.chart.data_series]
    assert "% GDP" in [series.name for series in result.chart.data_series]

def test_extract_with_loader_and_vision():
    # Arrange
    test_file_path = os.path.join(cwd, "tests", "files", "invoice.pdf")

    extractor = Extractor()
    loader = DocumentLoaderPyPdf()
    extractor.load_document_loader(loader)
    extractor.load_llm("gpt-4o-mini")

    content = loader.load(test_file_path)

    # Act
    result = extractor.extract(content, InvoiceContract, vision=True)

    # Assert
    assert result is not None
    assert result.invoice_number == "0000001"
    assert result.invoice_date == "2014-05-07"
    assert result.total_amount == 2500

    # Check line items
    assert len(result.lines) == 1
    line = result.lines[0]
    assert line.description == "Website Redesign"
    assert line.quantity == 1
    assert line.unit_price == 2500
    assert line.amount == 2500

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
import asyncio
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
    extractor.load_document_loader(
        DocumentLoaderPyPdf()
    )
    extractor.load_llm("gpt-4o-mini")
    # Act
    result = extractor.extract(test_file_path, InvoiceContract)

    # Assert
    assert result is not None
    assert result.lines[0].description == "Consultation services"
    assert result.lines[0].quantity == 3
    assert result.lines[0].unit_price == 375
    assert result.lines[0].amount == 1125

def test_vision_with_chart():
    # Arrange
    extractor = Extractor()
    extractor.load_llm("gpt-4o")
    test_file_path = os.path.join(cwd, "tests", "test_images", "image.png")

    # Act
    result = extractor.extract(test_file_path, ChartWithContent, vision=True)

    # Assert
    assert result is not None
    # TODO: For now is sanity to test for errors

def test_vision_content_pdf():
    # Arrange
    extractor = Extractor()
    extractor.load_llm("gpt-4o")
    test_file_path = os.path.join(cwd, "tests", "files", "invoice.pdf")

    # Act
    result = extractor.extract(test_file_path, InvoiceContract, vision=True)

    # Assert
    assert result is not None
    # TODO: For now is sanity to test for errors

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

if __name__ == "__main__":
    test_cancel_batch_extraction()
import asyncio
import os
import time
from dotenv import load_dotenv
from extract_thinker.extractor import Extractor
from extract_thinker.document_loader.document_loader_tesseract import DocumentLoaderTesseract
from tests.models.invoice import InvoiceContract
import pytest

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
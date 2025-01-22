import os
from typing import Optional
from dotenv import load_dotenv
from extract_thinker import DocumentLoaderPyPdf
from extract_thinker.document_loader.document_loader_docling import DocumentLoaderDocling, DoclingConfig
from extract_thinker import Extractor
from extract_thinker import Contract
from extract_thinker import Classification
from extract_thinker import DocumentLoaderMarkItDown
from extract_thinker.models.completion_strategy import CompletionStrategy
from extract_thinker import SplittingStrategy
from extract_thinker import Process
from extract_thinker import TextSplitter
from extract_thinker import ImageSplitter
from pydantic import Field

from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,
    TableStructureOptions,
)
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption

load_dotenv()
cwd = os.getcwd()

# Define the contracts as shown in the article
class InvoiceContract(Contract):
    invoice_number: str = Field(description="Unique invoice identifier")
    invoice_date: str = Field(description="Date of the invoice")
    total_amount: float = Field(description="Overall total amount")

class VehicleRegistration(Contract):
    name_primary: Optional[str] = Field(
        default=None,
        description="Primary registrant's name (Last, First, Middle)"
    )
    name_secondary: Optional[str] = Field(
        default=None,
        description="Co-registrant's name if applicable"
    )
    address: Optional[str] = Field(
        default=None,
        description="Primary registrant's mailing address including street, city, state and zip code"
    )
    vehicle_type: Optional[str] = Field(
        default=None,
        description="Type of vehicle (e.g., 2-Door, 4-Door, Pick-up, Van, etc.)"
    )
    vehicle_color: Optional[str] = Field(
        default=None,
        description="Primary color of the vehicle"
    )

class DriverLicenseContract(Contract):
    name: Optional[str] = Field(description="Full name on the license")
    age: Optional[int] = Field(description="Age of the license holder")
    license_number: Optional[str] = Field(description="License number")

def test_extract_with_ollama():
    test_file_path = os.path.join(cwd, "tests", "files", "invoice.pdf")

    extractor = Extractor()
    extractor.load_document_loader(
        DocumentLoaderPyPdf()
    )

    os.environ["API_BASE"] = "http://localhost:11434"
    extractor.load_llm("ollama/phi4")

    # Act
    result = extractor.extract(test_file_path, InvoiceContract)

    # Assert
    assert result is not None
    assert result.invoice_number == "00012"
    assert result.invoice_date == "1/30/23"

def test_extract_with_ollama_full_pipeline():
    """Test the complete document processing pipeline as described in the article"""
    # Setup test file path
    test_file_path = os.path.join(cwd, "tests", "files", "bulk.pdf")
    
    # Create classifications
    test_classifications = [
        Classification(
            name="Vehicle Registration",
            description="This is a vehicle registration document",
            contract=VehicleRegistration
        ),
        Classification(
            name="Driver License",
            description="This is a driver license document",
            contract=DriverLicenseContract
        )
    ]
    
    # Setup OCR options
    ocr_options = TesseractCliOcrOptions(
        force_full_page_ocr=True,
        tesseract_cmd="/opt/homebrew/bin/tesseract"
    )
    
    # Setup pipeline options
    pipeline_options = PdfPipelineOptions(
        do_table_structure=True,
        do_ocr=True,
        ocr_options=ocr_options,
        table_structure_options=TableStructureOptions(
            do_cell_matching=True
        )
    )
    
    # Create format options
    format_options = {
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }
    
    # Create docling config with OCR enabled
    docling_config = DoclingConfig(
        format_options=format_options,
        ocr_enabled=True,
        force_full_page_ocr=True
    )
    
    # Setup extractor with OCR-enabled docling loader
    extractor = Extractor()
    extractor.load_document_loader(DocumentLoaderDocling(docling_config))
    
    # Configure Ollama
    os.environ["API_BASE"] = "http://localhost:11434"
    extractor.load_llm("ollama/phi4")
    
    # Attach extractor to classifications
    for classification in test_classifications:
        classification.extractor = extractor
    
    # Setup process
    process = Process()
    process.load_document_loader(DocumentLoaderDocling(docling_config))
    process.load_splitter(ImageSplitter(model="claude-3-5-sonnet-20241022"))

    test_classifications[0].extractor = extractor
    test_classifications[1].extractor = extractor

    # Run the complete pipeline
    result = (
        process
        .load_file(test_file_path)
        .split(test_classifications, strategy=SplittingStrategy.LAZY)
        .extract(vision=False, completion_strategy=CompletionStrategy.PAGINATE)
    )
    
    # Assert
    assert result is not None
    assert isinstance(result, list)
    
    # Check each extracted item
    for item in result:
        assert isinstance(item, (VehicleRegistration, DriverLicenseContract))
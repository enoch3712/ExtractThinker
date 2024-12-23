from io import BytesIO
import os
from typing import List, Optional
from dotenv import load_dotenv
from extract_thinker import (
    Classification, 
    Extractor, 
    ImageSplitter, 
    Process, 
    SplittingStrategy,
    Contract,
    DocumentLoaderDocumentAI
)
from pydantic import BaseModel, field_validator

from extract_thinker.document_loader.document_loader_aws_textract import DocumentLoaderAWSTextract
from extract_thinker.document_loader.document_loader_azure_document_intelligence import DocumentLoaderAzureForm
from extract_thinker.document_loader.document_loader_pypdf import DocumentLoaderPyPdf
from extract_thinker.text_splitter import TextSplitter

# from extract_thinker.document_loader.document_loader_pypdf import DocumentLoaderPyPdf

load_dotenv()

# Define test constants
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BULK_DOC_PATH = os.path.join(CURRENT_DIR, "files", "bulk.pdf")
GOOGLE_CREDENTIALS_PATH = os.path.join(CURRENT_DIR, "credentials", "google_credentials.json")

class InvoiceLine(BaseModel):
    description: str
    quantity: int
    unit_price: float
    amount: float

    @field_validator('quantity', mode='before')
    def convert_quantity_to_int(cls, v):
        if isinstance(v, float):
            return int(v)
        return v

class VehicleRegistration(Contract):
    name_primary: str
    name_last: Optional[str]
    address: str
    vehicle_type: str
    vehicle_color: str

class DriverLicense(Contract):
    name: str
    age: int
    license_number: str

def setup_process_with_document_ai():
    """Helper function to set up process with Google Document AI"""
    # Set required environment variables
    os.environ["VERTEXAI_PROJECT"] = "extractthinker"
    os.environ["VERTEXAI_LOCATION"] = "us-central1"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/regina/Downloads/extractthinker-eb7d824a7f67.json"

    # Initialize document loader
    document_loader = DocumentLoaderDocumentAI(
        project_id="496372363784",
        location="eu",
        processor_id="9203063ba6e697ce",
        credentials=os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    )

    # document_loader = DocumentLoaderPyPdf()

    # document_loader = DocumentLoaderAzureForm(
    #     subscription_key=os.getenv("AZURE_SUBSCRIPTION_KEY"),
    #     endpoint=os.getenv("AZURE_ENDPOINT")
    # )

    # Convert file to BytesIO before passing to loader
    with open("tests/files/bulk.pdf", "rb") as file:
        bytes_content = BytesIO(file.read())
        content = document_loader.load(bytes_content)

    # Initialize extractor
    extractor = Extractor()
    extractor.load_document_loader(document_loader)
    extractor.load_llm("vertex_ai/gemini-2.0-flash-exp")

    # Create classifications
    classifications = [
        Classification(
            name="Vehicle Registration",
            description="This is a vehicle registration document",
            contract=VehicleRegistration,
            extractor=extractor
        ),
        Classification(
            name="Driver License",
            description="This is a driver license document",
            contract=DriverLicense,
            extractor=extractor
        )
    ]

    # Initialize process
    process = Process()
    process.load_document_loader(document_loader)
    process.load_splitter(ImageSplitter("vertex_ai/gemini-2.0-flash-exp"))

    return process, classifications

def test_document_ai_eager_splitting():
    """Test eager splitting strategy with Document AI loader"""
    # Arrange
    process, classifications = setup_process_with_document_ai()

    # Act
    result = process.load_file(BULK_DOC_PATH)\
        .split(classifications, strategy=SplittingStrategy.EAGER)\
        .extract(vision=True)

    # Assert
    assert result is not None
    for item in result:
        assert isinstance(item, (VehicleRegistration, DriverLicense))

    # Verify vehicle registration data
    assert result[0].name_primary == "Motorist, Michael M"
    
    # Verify driver license data
    assert result[1].age == 65
    assert result[1].license_number.replace(" ", "") in ["0123456789", "123456789"]

if __name__ == "__main__":
    test_document_ai_eager_splitting()

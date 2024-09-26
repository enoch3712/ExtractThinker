import os
from dotenv import load_dotenv

from extract_thinker.document_loader.document_loader_tesseract import DocumentLoaderTesseract
from extract_thinker.process import Process
from extract_thinker.image_splitter import ImageSplitter
from extractor import Extractor
from extract_thinker.models.classification import Classification
from tests.models.invoice import InvoiceContract
from tests.models.driver_license import DriverLicense

load_dotenv()

# Usage
extractor = Extractor()
extractor.load_document_loader(DocumentLoaderTesseract(os.getenv("TESSERACT_PATH")))
extractor.load_llm("gpt-3.5-turbo")

classifications = [
    Classification(name="Driver License", description="This is a driver license", contract=DriverLicense, extractor=extractor),
    Classification(name="Invoice", description="This is an invoice", contract=InvoiceContract, extractor=extractor)
]

process = Process()
process.load_document_loader(DocumentLoaderTesseract(os.getenv("TESSERACT_PATH")))
process.load_splitter(ImageSplitter())

path = "C:\\Users\\Lopez\\Desktop\\MagniFinance\\examples\\outputTestOne.pdf"
other_path = "C:\\Users\\Lopez\\Desktop\\MagniFinance\\examples\\SingleInvoiceTests\\FT63O.pdf"

split_content = process.load_file(path)\
    .split(classifications)\
    .extract()

# extractor.loadSplitter(ImageSplitter())
# extractor.loadfile(
#     "C:\\Users\\Lopez\\Desktop\\MagniFinance\\examples\\outputTestOne.pdf"
# )
# extractor.split(classifications)

# extractor.loadfile("C:\\Users\\Lopez\\Desktop\\MagniFinance\\examples\\outputTestOne.pdf").split(classifications)

extractor.load_document_loader(
    DocumentLoaderTesseract(os.getenv("TESSERACT_PATH"))
)
extractor.load_llm("claude-3-haiku-20240307")

# extractor.classify_from_path(
#     "C:\\Users\\Lopez\\Desktop\\ExtractThinker\\driverLicense.jpg",
#     classifications
# )

# extractor.loadfile(
#     "C:\\Users\\Lopez\\Desktop\\ExtractThinker\\driverLicense.jpg"
#     )\
#     .split(classifications)\
#     .extract()\
#     .where(lambda x: x.name == "Driver License")\

# user_info = extractor.extract_from_file(
#     'C:\\Users\\Lopez\\Desktop\\ExtractThinker\\driverLicense.jpg', UserContract, vision=True)

# print(user_info.name)
# print(user_info.age)

# the equivalent of this for the instructor:

# equivalent for this, inside instructor: json.loads(json_string)
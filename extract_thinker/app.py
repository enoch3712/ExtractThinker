from dotenv import load_dotenv

from extract_thinker.document_loader.document_loader_tesseract import DocumentLoaderTesseract
from extractor import Extractor
from models import Classification


load_dotenv()

classifications = [
    Classification(name="Driver License", description="This is a driver license"),
    Classification(name="Invoice", description="This is an invoice"),
]

# Usage
extractor = Extractor()

# extractor.loadSplitter(ImageSplitter())
# extractor.loadfile(
#     "C:\\Users\\Lopez\\Desktop\\MagniFinance\\examples\\outputTestOne.pdf"
# )
# extractor.split(classifications)

# extractor.loadfile("C:\\Users\\Lopez\\Desktop\\MagniFinance\\examples\\outputTestOne.pdf").split(classifications)

extractor.load_document_loader(
    DocumentLoaderTesseract("C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
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
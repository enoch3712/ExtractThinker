import os

from dotenv import load_dotenv

from extract_thinker import DocumentLoaderTesseract, Extractor, Contract

load_dotenv()
cwd = os.getcwd()


class InvoiceContract(Contract):
    invoice_number: str
    invoice_date: str


tesseract_path = os.getenv("TESSERACT_PATH")
test_file_path = os.path.join(cwd, "tests", "test_images", "invoice.png")

extractor = Extractor()
extractor.load_document_loader(
    DocumentLoaderTesseract(tesseract_path)
)
extractor.load_llm("claude-3-haiku-20240307")

result = extractor.extract(test_file_path, InvoiceContract)

if result is not None:
    print("Extraction successful.")
else:
    print("Extraction failed.")

print("Invoice Number: ", result.invoice_number)
print("Invoice Date: ", result.invoice_date)

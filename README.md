# ExtractThinker

Library to extract data from files and documents agnostically using LLMs. `extract_thinker` provides ORM-style interaction between files and LLMs, allowing for flexible and powerful document extraction workflows.

## Features

- Supports multiple document loaders including Tesseract OCR, Azure Form Recognizer, AWS TextExtract, Google Document AI.
- Customizable extraction using contract definitions.
- Asynchronous processing for efficient document handling.
- Built-in support for various document formats.
- ORM-style interaction between files and LLMs.

## Installation

To install `extract_thinker`, you can use `pip`:

```bash
pip install extract_thinker
```

## Usage
Here's a quick example to get you started with extract_thinker. This example demonstrates how to load a document using Tesseract OCR and extract specific fields defined in a contract.

```python
import os
from dotenv import load_dotenv
from extract_thinker import DocumentLoaderTesseract, Extractor, Contract

load_dotenv()
cwd = os.getcwd()

class InvoiceContract(Contract):
    invoice_number: str
    invoice_date: str

# Arrange
tesseract_path = os.getenv("TESSERACT_PATH")
test_file_path = os.path.join(cwd, "test_images", "invoice.png")

extractor = Extractor()
extractor.load_document_loader(
    DocumentLoaderTesseract(tesseract_path)
)
extractor.load_llm("claude-3-haiku-20240307")

# Act
result = extractor.extract(test_file_path, InvoiceContract)

# Assert
assert result is not None
assert result.invoice_number == "0000001"
assert result.invoice_date == "2014-05-07"
```

## Splitting Files Example
You can also split and process documents using extract_thinker. Here's how you can do it:

```python
import os
from dotenv import load_dotenv
from extract_thinker import DocumentLoaderTesseract, Extractor, Process, Classification, ImageSplitter

load_dotenv()

class DriverLicense(Contract):
    # Define your DriverLicense contract fields here
    pass

class InvoiceContract(Contract):
    invoice_number: str
    invoice_date: str

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

# Process the split_content as needed
```

## Additional Examples
You can find more examples in the repository. These examples cover various use cases and demonstrate the flexibility of extract_thinker.

## Contributing
We welcome contributions from the community! If you would like to contribute, please follow these steps:

Fork the repository.
Create a new branch for your feature or bugfix.
Write tests for your changes.
Run tests to ensure everything is working correctly.
Submit a pull request with a description of your changes.
License
This project is licensed under the Apache License 2.0. See the LICENSE file for more details.

## Contact
For any questions or issues, please open an issue on the GitHub repository.
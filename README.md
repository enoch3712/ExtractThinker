<p align="center">
  <img src="https://github.com/enoch3712/Open-DocLLM/assets/9283394/41d9d151-acb5-44da-9c10-0058f76c2512" alt="Extract Thinker Logo" width="200"/>
</p>
<p align="center">
<img alt="Python Version" src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" />
<a href="https://medium.com/@enoch3712">
    <img alt="Medium" src="https://img.shields.io/badge/Medium-12100E?style=flat&logo=medium&logoColor=white" />
</a>
<img alt="GitHub Last Commit" src="https://img.shields.io/github/last-commit/enoch3712/Open-DocLLM" />
<img alt="Github License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" />
</p>

# ExtractThinker

Library to extract data from files and documents agnostically using LLMs. `extract_thinker` provides ORM-style interaction between files and LLMs, allowing for flexible and powerful document extraction workflows.

## Features

- Supports multiple document loaders including Tesseract OCR, Azure Form Recognizer, AWS TextExtract, Google Document AI.
- Customizable extraction using contract definitions.
- Asynchronous processing for efficient document handling.
- Built-in support for various document formats.
- ORM-style interaction between files and LLMs.

<p align="center">
  <img src="https://github.com/enoch3712/Open-DocLLM/assets/9283394/b1b8800c-3c55-4ee5-92fe-b8b663c7a81f" alt="Extract Thinker Features Diagram" width="300"/>
</p>

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

tesseract_path = os.getenv("TESSERACT_PATH")
test_file_path = os.path.join(cwd, "test_images", "invoice.png")

extractor = Extractor()
extractor.load_document_loader(
    DocumentLoaderTesseract(tesseract_path)
)
extractor.load_llm("claude-3-haiku-20240307")

result = extractor.extract(test_file_path, InvoiceContract)

print("Invoice Number: ", result.invoice_number)
print("Invoice Date: ", result.invoice_date)
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

path = "..."

split_content = process.load_file(path)\
    .split(classifications)\
    .extract()

# Process the split_content as needed
```

## Infrastructure

The `extract_thinker` project is inspired by the LangChain ecosystem, featuring a modular infrastructure with templates, components, and core functions to facilitate robust document extraction and processing. 

<p align="center">
  <img src="https://github.com/enoch3712/Open-DocLLM/assets/9283394/996fb2de-0558-4f13-ab3d-7ea56a593951" alt="Extract Thinker Logo" width="400"/>
</p>

## 📖 Examples

| Notebook | Description |
|----------|-------------|
| [Basic Usage](examples/notebooks/basic_example.ipynb) | Basic usage of ExtractThinker with PyPDF loader and GPT-4o-mini for invoice data extraction |

## Why Just Not LangChain?
While LangChain is a generalized framework designed for a wide array of use cases, extract_thinker is specifically focused on Intelligent Document Processing (IDP). Although achieving 100% accuracy in IDP remains a challenge, leveraging LLMs brings us significantly closer to this goal.

## Additional Examples
You can find more examples in the repository. These examples cover various use cases and demonstrate the flexibility of extract_thinker. Also check my the medium of the author that contains several examples about the library

## Contributing
We welcome contributions from the community! If you would like to contribute, please follow these steps:

Fork the repository.
Create a new branch for your feature or bugfix.
Write tests for your changes.
Run tests to ensure everything is working correctly.
Submit a pull request with a description of your changes.

## Community
Júlio Almeida
    https://pub.towardsai.net/extractthinker-ai-document-intelligence-with-llms-72cbce1890ef

## License
This project is licensed under the Apache License 2.0. See the LICENSE file for more details.

## Contact
For any questions or issues, please open an issue on the GitHub repository.
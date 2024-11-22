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

ExtractThinker is a flexible document intelligence tool that leverages Large Language Models (LLMs) to extract and classify structured data from documents, functioning like an ORM for seamless document processing workflows.

**TL;DR Document Intelligence for LLMs**

## 🚀 Key Features

- **Flexible Document Loaders**: Support for multiple document loaders, including Tesseract OCR, Azure Form Recognizer, AWS Textract, Google Document AI, and more.
- **Customizable Contracts**: Define custom extraction contracts using Pydantic models for precise data extraction.
- **Advanced Classification**: Classify documents or document sections using custom classifications and strategies.
- **Asynchronous Processing**: Utilize asynchronous processing for efficient handling of large documents.
- **Multi-format Support**: Seamlessly work with various document formats like PDFs, images, spreadsheets, and more.
- **ORM-style Interaction**: Interact with documents and LLMs in an ORM-like fashion for intuitive development.
- **Splitting Strategies**: Implement lazy or eager splitting strategies to process documents page by page or as a whole.
- **Integration with LLMs**: Easily integrate with different LLM providers like OpenAI, Anthropic, Cohere, and more.
- **Community-driven Development**: Inspired by the LangChain ecosystem with a focus on intelligent document processing.
![image](https://github.com/user-attachments/assets/844b425c-0bb7-4abc-9d08-96e4a736d096)

## 📦 Installation

Install ExtractThinker using pip:

```bash
pip install extract_thinker
```

## 🛠️ Usage

### Basic Extraction Example

Here's a quick example to get you started with ExtractThinker. This example demonstrates how to load a document using PyPdf and extract specific fields defined in a contract.

```python
import os
from dotenv import load_dotenv
from extract_thinker import Extractor, DocumentLoaderPyPdf, Contract

load_dotenv()

class InvoiceContract(Contract):
    invoice_number: str
    invoice_date: str

# Set the path to your Tesseract executable
test_file_path = os.path.join("path_to_your_files", "invoice.pdf")

# Initialize the extractor
extractor = Extractor()
extractor.load_document_loader(DocumentLoaderPyPdf())
extractor.load_llm("gpt-4o-mini")  # or any other supported model

# Extract data from the document
result = extractor.extract(test_file_path, InvoiceContract)

print("Invoice Number:", result.invoice_number)
print("Invoice Date:", result.invoice_date)
```

### Classification Example

ExtractThinker allows you to classify documents or parts of documents using custom classifications:

```python
import os
from dotenv import load_dotenv
from extract_thinker import (
    Extractor, Classification, Process, ClassificationStrategy,
    DocumentLoaderPyPdf, Contract
)

load_dotenv()

class InvoiceContract(Contract):
    invoice_number: str
    invoice_date: str

class DriverLicenseContract(Contract):
    name: str
    license_number: str

# Initialize the extractor and load the document loader
extractor = Extractor()
extractor.load_document_loader(DocumentLoaderPyPdf())
extractor.load_llm("gpt-4o-mini")

# Define classifications
classifications = [
    Classification(
        name="Invoice",
        description="An invoice document",
        contract=InvoiceContract,
        extractor=extractor,
    ),
    Classification(
        name="Driver License",
        description="A driver's license document",
        contract=DriverLicenseContract,
        extractor=extractor,
    ),
]

# Classify the document directly using the extractor
result = extractor.classify(
    "path_to_your_document.pdf",  # Can be a file path or IO stream
    classifications,
    image=True  # Set to True for image-based classification
)

# The result will be a ClassificationResponse object with 'name' and 'confidence' fields
print(f"Document classified as: {result.name}")
print(f"Confidence level: {result.confidence}")
```

### Splitting Files Example

ExtractThinker allows you to split and process documents using different strategies. Here's how you can split a document and extract data based on classifications.

```python
import os
from dotenv import load_dotenv
from extract_thinker import (
    Extractor,
    Process,
    Classification,
    ImageSplitter,
    DocumentLoaderTesseract,
    Contract,
    SplittingStrategy,
)

load_dotenv()

class DriverLicenseContract(Contract):
    name: str
    license_number: str

class InvoiceContract(Contract):
    invoice_number: str
    invoice_date: str

# Initialize the extractor and load the document loader
extractor = Extractor()
extractor.load_document_loader(DocumentLoaderPyPdf())
extractor.load_llm("gpt-4o-mini")

# Define classifications
classifications = [
    Classification(
        name="Driver License",
        description="A driver's license document",
        contract=DriverLicenseContract,
        extractor=extractor,
    ),
    Classification(
        name="Invoice",
        description="An invoice document",
        contract=InvoiceContract,
        extractor=extractor,
    ),
]

# Initialize the process and load the splitter
process = Process()
process.load_document_loader(DocumentLoaderPyPdf())
process.load_splitter(ImageSplitter(model="gpt-4o-mini"))

# Load and process the document
path_to_document = "path_to_your_multipage_document.pdf"
split_content = (
    process.load_file(path_to_document)
    .split(classifications, strategy=SplittingStrategy.LAZY)
    .extract()
)

# Process the extracted content as needed
for item in split_content:
    if isinstance(item, InvoiceContract):
        print("Extracted Invoice:")
        print("Invoice Number:", item.invoice_number)
        print("Invoice Date:", item.invoice_date)
    elif isinstance(item, DriverLicenseContract):
        print("Extracted Driver License:")
        print("Name:", item.name)
        print("License Number:", item.license_number)

```

### Batch Processing Example

You can also perform batch processing of documents:

```python
from extract_thinker import Extractor, Contract

class ReceiptContract(Contract):
    store_name: str
    total_amount: float

extractor = Extractor()
extractor.load_llm("gpt-4o-mini")

# List of file paths or streams
document = "receipt1.jpg"

batch_job = extractor.extract_batch(
    source=document,
    response_model=ReceiptContract,
    vision=True,
)

# Monitor the batch job status
print("Batch Job Status:", await batch_job.get_status())

# Retrieve results once processing is complete
results = await batch_job.get_result()
for result in results.parsed_results:
    print("Store Name:", result.store_name)
    print("Total Amount:", result.total_amount)
```

### Local LLM Integration Example

ExtractThinker supports custom LLM integrations. Here's how you can use a custom LLM:

```python
from extract_thinker import Extractor, LLM, DocumentLoaderTesseract, Contract

class InvoiceContract(Contract):
    invoice_number: str
    invoice_date: str

# Initialize the extractor
extractor = Extractor()
extractor.load_document_loader(DocumentLoaderTesseract(os.getenv("TESSERACT_PATH")))

# Load a custom LLM (e.g., Ollama)
llm = LLM('ollama/phi3', api_base='http://localhost:11434')
extractor.load_llm(llm)

# Extract data
result = extractor.extract("invoice.png", InvoiceContract)
print("Invoice Number:", result.invoice_number)
print("Invoice Date:", result.invoice_date)
```

## 📚 Documentation and Resources

- **Examples**: Check out the examples directory for Jupyter notebooks and scripts demonstrating various use cases.
- **Medium Articles**: Read articles about ExtractThinker on the author's Medium page.
- **Test Suite**: Explore the test suite in the tests/ directory for more advanced usage examples and test cases.

## 🧩 Integration with LLM Providers

ExtractThinker supports integration with multiple LLM providers:

- **OpenAI**: Use models like gpt-3.5-turbo, gpt-4, etc.
- **Anthropic**: Integrate with Claude models.
- **Cohere**: Utilize Cohere's language models.
- **Azure OpenAI**: Connect with Azure's OpenAI services.
- **Local Models**: Ollama compatible models.

## ⚙️ How It Works

ExtractThinker uses a modular architecture inspired by the LangChain ecosystem:

- **Document Loaders**: Responsible for loading and preprocessing documents from various sources and formats.
- **Extractors**: Orchestrate the interaction between the document loaders and LLMs to extract structured data.
- **Splitters**: Implement strategies to split documents into manageable chunks for processing.
- **Contracts**: Define the expected structure of the extracted data using Pydantic models.
- **Classifications**: Classify documents or document sections to apply appropriate extraction contracts.
- **Processes**: Manage the workflow of loading, classifying, splitting, and extracting data from documents.

![image](https://github.com/user-attachments/assets/b12ba937-20a8-47da-a778-c126bc1748b3)

## 📝 Why Use ExtractThinker?

While general frameworks like LangChain offer a broad range of functionalities, ExtractThinker is specialized for Intelligent Document Processing (IDP). It simplifies the complexities associated with IDP by providing:

- **Specialized Components**: Tailored tools for document loading, splitting, and extraction.
- **High Accuracy with LLMs**: Leverages the power of LLMs to improve the accuracy of data extraction and classification.
- **Ease of Use**: Intuitive APIs and ORM-style interactions reduce the learning curve.
- **Community Support**: Active development and support from the community.

## 🤝 Contributing

We welcome contributions from the community! To contribute:

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Write tests for your changes
4. Run tests to ensure everything is working correctly
5. Submit a pull request with a description of your changes

## 🌟 Community and Support

Stay updated and connect with the community:
- [Scaling Document Extraction with o1, GPT-4o & Mini](https://medium.com/towards-artificial-intelligence/scaling-document-extraction-with-o1-gpt4o-and-mini-extractthinker-8f3340b4e69c)
- [Claude 3.5 — The King of Document Intelligence](https://medium.com/gitconnected/claude-3-5-the-king-of-document-intelligence-f57bea1d209d?sk=124c5abb30c0e7f04313c5e20e79c2d1)
- [Classification Tree for LLMs](https://medium.com/gitconnected/classification-tree-for-llms-32b69015c5e0?sk=8a258cf74fe3483e68ab164e6b3aaf4c)
- [Advanced Document Classification with LLMs](https://medium.com/gitconnected/advanced-document-classification-with-llms-8801eaee3c58?sk=f5a22ee72022eb70e112e3e2d1608e79)
- [Phi-3 and Azure: PDF Data Extraction | ExtractThinker](https://medium.com/towards-artificial-intelligence/phi-3-and-azure-pdf-data-extraction-extractthinker-cb490a095adb?sk=7be7e625b8f9932768442f87dd0ebcec)
- [ExtractThinker: Document Intelligence for LLMs](https://medium.com/towards-artificial-intelligence/extractthinker-ai-document-intelligence-with-llms-72cbce1890ef)

## 📄 License

This project is licensed under the Apache License 2.0. See the LICENSE file for more details.

## Contact

For any questions or issues, please open an issue on the GitHub repository or reach out via email.

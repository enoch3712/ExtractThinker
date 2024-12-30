# Microsoft Word Document Loader (Doc2txt)

The Doc2txt loader is designed to handle Microsoft Word documents (`.doc` and `.docx` files). It uses the `docx2txt` library to extract text content from Word documents.

## Basic Usage

```python
from extract_thinker import Extractor, DocumentLoaderDoc2txt

# Initialize the extractor with Doc2txt loader
extractor = Extractor()
extractor.load_document_loader(DocumentLoaderDoc2txt())

# Process a Word document
result = extractor.extract("document.docx", YourContract)
```

## Features

- Supports both `.doc` and `.docx` file formats
- Automatically splits content into pages using double newlines as separators
- Preserves text formatting and structure
- Caches results for improved performance

## Limitations

- Does not support vision mode (images within Word documents are not processed)
- Does not preserve complex formatting or document styling
- Tables and other structured content may lose their layout
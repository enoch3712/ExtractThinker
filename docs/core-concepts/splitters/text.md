# Text Splitter

Text Splitter is designed to handle text-based document splitting by analyzing content continuity and relationships between pages.

## Basic Usage

```python
from extract_thinker import TextSplitter, Process, SplittingStrategy
from extract_thinker.document_loader import DocumentLoaderTesseract

# Initialize process and loader
process = Process()
process.load_document_loader(DocumentLoaderTesseract(tesseract_path))

# Initialize text splitter with model
process.load_splitter(TextSplitter("claude-3-5-sonnet-20241022"))

# Split document
result = process.load_file("document.pdf")\
    .split(classifications, strategy=SplittingStrategy.EAGER)\
    .extract()
```
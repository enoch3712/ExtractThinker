# Image Splitter

Image Splitter is specialized for handling image-based document splitting by analyzing visual consistency and layout patterns between pages.

## Basic Usage

```python
from extract_thinker import ImageSplitter, Process, SplittingStrategy
from extract_thinker.document_loader import DocumentLoaderTesseract

# Initialize process and loader
process = Process()
process.load_document_loader(DocumentLoaderTesseract(tesseract_path))

# Initialize image splitter with vision model
process.load_splitter(ImageSplitter("claude-3-5-sonnet-20241022"))

# Split document
result = process.load_file("document.pdf")\
    .split(classifications, strategy=SplittingStrategy.EAGER)\
    .extract()
```

For more examples and advanced usage, check out the [examples directory](examples/) in the repository. 
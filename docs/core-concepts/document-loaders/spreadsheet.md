# Spreadsheet Document Loader

The Spreadsheet loader in ExtractThinker handles Excel, CSV, and other tabular data formats.

## Basic Usage

Here's how to use the Spreadsheet loader:

```python
from extract_thinker import Extractor
from extract_thinker.document_loader import DocumentLoaderSpreadsheet

# Initialize the loader
loader = DocumentLoaderSpreadsheet()

# Load Excel file
excel_content = loader.load_content_from_file("data.xlsx")

# Load CSV file
csv_content = loader.load_content_from_file("data.csv")
```

## Features

- Excel file support (.xlsx, .xls)
- CSV file support
- Multiple sheet handling
- Data type preservation

## Best Practices

1. **Data Preparation**
   - Use consistent data formats
   - Clean data before processing
   - Handle missing values appropriately

2. **Performance**
   - Process large files in chunks
   - Use appropriate data types
   - Consider memory limitations

For more examples and implementation details, check out the [examples directory](examples/) in the repository. 
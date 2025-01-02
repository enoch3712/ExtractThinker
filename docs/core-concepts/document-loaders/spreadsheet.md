# Spreadsheet Document Loader

The spreadsheet loader is designed to handle various spreadsheet formats including Excel files (xls, xlsx, xlsm, xlsb) and OpenDocument formats (odf, ods, odt).

## Installation

To use the spreadsheet loader, you need to install the required dependencies:

```bash
pip install openpyxl xlrd
```

## Supported Formats

`xls`, `xlsx`, `xlsm`, `xlsb`, `odf`, `ods`, `odt`, `csv`

## Usage

```python
from extract_thinker import DocumentLoaderSpreadSheet

# Initialize the loader
loader = DocumentLoaderSpreadSheet()

# Load from file
pages = loader.load("path/to/your/spreadsheet.xlsx")

# Load CSV file
csv_content = loader.load_content_from_file("data.csv")
```

## Features

- Excel file support (.xlsx, .xls)
- CSV file support
- Multiple sheet handling
- Data type preservation

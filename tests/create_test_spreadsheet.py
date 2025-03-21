import os
import pandas as pd

# Create a directory for test files if it doesn't exist
os.makedirs('tests/files', exist_ok=True)

# Create a DataFrame with test data
data = {
    'Column1': ['test_value', 'data2', 'data3'],
    'Column2': [10, 20, 30],
    'Column3': ['A', 'B', 'C']
}
df1 = pd.DataFrame(data)

# Create a second sheet with different data
data2 = {
    'Name': ['John', 'Alice', 'Bob'],
    'Age': [25, 30, 35],
    'Location': ['New York', 'Paris', 'London']
}
df2 = pd.DataFrame(data2)

# Create Excel writer object
file_path = 'tests/files/test_spreadsheet.xlsx'
with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
    # Write each DataFrame to a different sheet
    df1.to_excel(writer, sheet_name='Sheet1', index=False)
    df2.to_excel(writer, sheet_name='Sheet2', index=False)

print(f"Test Excel file created at {file_path}") 
import os
import pytest
from dotenv import load_dotenv

from extract_thinker.extractor import Extractor
from extract_thinker.process import MaskingStrategy, Process
from extract_thinker.document_loader.document_loader_pypdf import DocumentLoaderPyPdf
from extract_thinker.llm import LLM
import asyncio

load_dotenv()
cwd = os.getcwd()

def test_mask():
    # Arrange
    test_file_path = os.path.join(cwd, "tests", "files", "invoice.pdf")

    process = Process()
    process.load_document_loader(DocumentLoaderPyPdf())
    process.load_file(test_file_path)
    process.add_masking_llm("groq/llama-3.2-3b-preview")

    # Act
    test_text = (
        "Mr. George Collins lives at 123 Main St, Anytown, USA 12345. His phone number is 555-1234. "
        "Jane Smith resides at 456 Elm Avenue, Othercity, State 67890, and can be reached at (987) 654-3210. "
        "The company's CEO, Robert Johnson, has an office at 789 Corporate Blvd, Suite 500, Bigcity, State 13579. "
        "For customer service, call 1-800-555-9876 or email support@example.com. "
        "Sarah Lee, our HR manager, can be contacted at 444-333-2222 or sarah.lee@company.com. "
        "The project budget is $250,000, with an additional $50,000 allocated for contingencies. "
        "Monthly maintenance costs are estimated at $3,500. "
        "For international clients, please use +1-555-987-6543. "
        "Our tax ID number is 12-3456789."
    )

    result = asyncio.run(process.mask_content(test_text))

    # Assert
    assert result.masked_text is not None
    assert result.mapping is not None

    # Check if all original sensitive information is masked
    sensitive_info = [
        "George Collins", "123 Main St", "555-1234",
        "Jane Smith", "456 Elm Avenue", "(987) 654-3210",
        "Robert Johnson", "789 Corporate Blvd", 
        "1-800-555-9876", "support@example.com",
        "Sarah Lee", "444-333-2222", "sarah.lee@company.com",
        "$250,000", "$50,000", "$3,500",
        "+1-555-987-6543", "12-3456789"
    ]
    for info in sensitive_info:
        assert info not in result.masked_text, f"{info} was not masked properly"

    # Check if placeholders are present in masked text
    placeholder_types = ["NAME", "ADDRESS", "PHONE", "EMAIL"]
    assert any(f"[{type}" in result.masked_text for type in placeholder_types), "No expected placeholders found in masked text"

    # Check mapping
    assert len(result.mapping) >= 10, "Mapping should contain at least 10 items"
    assert all(key.startswith('[') and key.endswith(']') for key in result.mapping.keys()), "Mapping keys should be enclosed in square brackets"
    assert all(isinstance(value, str) for value in result.mapping.values()), "Mapping values should be strings"

    # Test unmasking
    unmasked_content = process.unmask_content(result.masked_text, result.mapping)
    assert "George Collins" in unmasked_content, "Unmasking failed for 'George Collins'"
    assert "123 Main St" in unmasked_content, "Unmasking failed for '123 Main St'"
    assert "555-1234" in unmasked_content, "Unmasking failed for '555-1234'"

    # Check if all masked content is unmasked
    for placeholder, original in result.mapping.items():
        assert original in unmasked_content, f"Unmasking failed for {original}"
        assert placeholder not in unmasked_content, f"Placeholder {placeholder} still present in unmasked content"

# def test_mask_invoice():
#     # Arrange
#     test_file_path = os.path.join(cwd, "tests", "files", "invoice.pdf")

#     process = Process()
#     process.load_document_loader(DocumentLoaderPyPdf())
#     process.load_file(test_file_path)
#     process.add_masking_llm("groq/llama-3.2-11b-text-preview")

#     # Act
#     test_text = (
#         "Market Financial Consulting   Experts in earning trusts   450 East 78th Ave  Denver, CO 12345   "
#         "Phone : (123) 456 -7890 Fax: (123) 456 -7891 INVOICE  INVOICE:  00012  DATE:  1/30/23  "
#         "TO: Gaurav Cheema   Caneiro Group   89 Pacific Ave   San Francisco, CA 78910   "
#         "FOR:  Consultation services    DESCRIPTION   HOURS  RATE  AMOUNT  "
#         "Consultation services   3.0  375.00   1125.00                                                                   "
#         "TOTAL  1125.00    Make all checks payable to  Market Financial Consulting  "
#         "Total due in 15 days. Overdue accounts subject to a service charge of 1% per month.   "
#         "THANK YOU FOR YOUR BUSINESS!"
#     )

#     result = asyncio.run(process.mask_content(test_text))

#     # Assert
#     assert result.masked_text is not None
#     assert result.mapping is not None

#     # Check if all original sensitive information is masked
#     sensitive_info = [
#         "Market Financial Consulting", "450 East 78th Ave", "Denver, CO 12345",
#         "(123) 456 -7890", "(123) 456 -7891", 
#         "Gaurav Cheema", "Caneiro Group", "89 Pacific Ave", "San Francisco, CA 78910"
#     ]
#     for info in sensitive_info:
#         assert info not in result.masked_text, f"{info} was not masked properly"

#     # Check if placeholders are present in masked text
#     placeholder_types = ["COMPANY", "ADDRESS", "PHONE", "NAME"]
#     assert any(f"[{type}" in result.masked_text for type in placeholder_types), "No expected placeholders found in masked text"

#     # Check if non-sensitive information is still present
#     non_sensitive_info = [
#         "INVOICE", "DATE", "FOR:  Consultation services", 
#         "DESCRIPTION", "HOURS", "RATE", "AMOUNT",
#         "3.0", "375.00", "1125.00", "TOTAL", 
#         "Make all checks payable to", 
#         "Total due in 15 days. Overdue accounts subject to a service charge of 1% per month.",
#         "THANK YOU FOR YOUR BUSINESS!"
#     ]
#     for info in non_sensitive_info:
#         assert info in result.masked_text, f"{info} was unexpectedly masked"

#     # Check mapping
#     assert len(result.mapping) >= 5, "Mapping should contain at least 5 items"
#     assert all(key.startswith('[') and key.endswith(']') for key in result.mapping.keys()), "Mapping keys should be enclosed in square brackets"
#     assert all(isinstance(value, str) for value in result.mapping.values()), "Mapping values should be strings"

#     # Test unmasking
#     unmasked_content = process.unmask_content(result.masked_text, result.mapping)
#     assert "Market Financial Consulting" in unmasked_content, "Unmasking failed for 'Market Financial Consulting'"
#     assert "450 East 78th Ave" in unmasked_content, "Unmasking failed for '450 East 78th Ave'"
#     assert "Gaurav Cheema" in unmasked_content, "Unmasking failed for 'Gaurav Cheema'"

#     # Check if all masked content is unmasked
#     for placeholder, original in result.mapping.items():
#         assert original in unmasked_content, f"Unmasking failed for {original}"
#         assert placeholder not in unmasked_content, f"Placeholder {placeholder} still present in unmasked content"

#     # Check if all masked content is the same as the original content
#     assert test_text == unmasked_content, "Unmasked content does not match the original content"

if __name__ == "__main__":
    test_mask()
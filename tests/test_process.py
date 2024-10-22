import os
import pytest
from dotenv import load_dotenv

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from extract_thinker.process import Process
from extract_thinker.document_loader.document_loader_pypdf import DocumentLoaderPyPdf
import asyncio

load_dotenv()
cwd = os.getcwd()

def test_mask():
    # Arrange
    test_file_path = os.path.join(cwd, "tests", "files", "invoice.pdf")

    process = Process()
    process.load_document_loader(DocumentLoaderPyPdf())
    process.load_file(test_file_path)
    process.add_masking_llm("groq/llama-3.2-11b-text-preview")

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

    # Act
    result = asyncio.run(process.mask_content(test_text))

    # Assert
    assert result.masked_text is not None
    assert result.mapping is not None

    # Check if all original PII is masked
    pii_info = {
        "persons": ["George Collins", "Jane Smith", "Robert Johnson", "Sarah Lee"],
        "addresses": [
            "123 Main St, Anytown, USA 12345",
            "456 Elm Avenue, Othercity, State 67890",
            "789 Corporate Blvd, Suite 500, Bigcity, State 13579",
        ],
        "phones": ["555-1234", "(987) 654-3210", "1-800-555-9876", "444-333-2222", "+1-555-987-6543"],
        "emails": ["support@example.com", "sarah.lee@company.com"],
        "tax_id": ["12-3456789"],
    }

    non_pii_info = [
        "Monthly maintenance costs are estimated at $3,500.",
    ]

    # Ensure PII is masked
    for person in pii_info["persons"]:
        assert person not in result.masked_text, f"PII {person} was not masked properly"

    for address in pii_info["addresses"]:
        assert address not in result.masked_text, f"PII address {address} was not masked properly"

    for phone in pii_info["phones"]:
        assert phone not in result.masked_text, f"PII phone {phone} was not masked properly"

    for email in pii_info["emails"]:
        assert email not in result.masked_text, f"PII email {email} was not masked properly"

    for tax in pii_info["tax_id"]:
        assert tax not in result.masked_text, f"PII tax ID {tax} was not masked properly"

    # Ensure non-PII data remains unchanged
    for info in non_pii_info:
        assert info in result.masked_text, f"Non-PII {info} was unexpectedly masked"

    # check if mapping length is 15
    assert len(result.mapping) == 15, "Mapping should contain 15 items"

    # Test unmasking
    unmasked_content = process.unmask_content(result.masked_text, result.mapping)

    # Optionally, verify the entire unmasked content matches the original
    assert unmasked_content == test_text, "Unmasked content does not match the original content"

def test_simple_use_case():
    # Arrange
    test_file_path = os.path.join(cwd, "tests", "files", "invoice.pdf")

    process = Process()
    process.load_document_loader(DocumentLoaderPyPdf())
    process.load_file(test_file_path)
    process.add_masking_llm("groq/llama-3.2-11b-text-preview")

    # Arrange
    test_text = "John Doe transferred $5000 to Jane Smith on 2021-05-01."

    # Act
    result = asyncio.run(process.mask_content(test_text))

    # Assert
    assert result.masked_text is not None
    assert result.mapping is not None

    # Ensure PII is masked
    assert "John Doe" not in result.masked_text
    assert "Jane Smith" not in result.masked_text

    # Ensure non-PII data remains
    assert "$5000" in result.masked_text
    assert "2021-05-01" in result.masked_text
    assert "transferred" in result.masked_text

    # Check mapping
    assert len(result.mapping) == 2
    assert "[PERSON1]" in result.mapping
    assert "[PERSON2]" in result.mapping
    assert result.mapping["[PERSON1]"] == "John Doe"
    assert result.mapping["[PERSON2]"] == "Jane Smith"

    # Test unmasking
    unmasked_content = process.unmask_content(result.masked_text, result.mapping)
    assert unmasked_content == test_text

if __name__ == "__main__":
    asyncio.run(test_simple_use_case())
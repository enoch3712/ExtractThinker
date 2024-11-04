from extract_thinker.models.contract import Contract
from tests.models.Page import Page
from pydantic import Field
from typing import List


class DocumentContent(Contract):
    classification: str = Field(description="The classification of the document, can be 'invoice', 'contract', 'report'")
    pages: List[Page] = Field(description="The pages of the document")
from typing import List
from pydantic import Field
from extract_thinker.models.contract import Contract

class PageContract(Contract):
    title: str
    number: int
    content: str = Field(description="Give me all the content, word for word")

class ReportContract(Contract):
    title: str
    pages: List[PageContract] 
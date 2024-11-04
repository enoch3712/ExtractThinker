from extract_thinker.models.contract import Contract
from tests.models.Chart import Chart
from pydantic import Field
from typing import List


class Page(Contract):
    content: str = Field(description="The content of the page without the chart")
    charts: List[Chart] = Field(description="The charts present in the page")
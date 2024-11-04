from extract_thinker.models.contract import Contract
from tests.models.Chart import Chart
from pydantic import Field


class ChartWithContent(Contract):
    content: str = Field(description="The content of the page without the chart")
    chart: Chart = Field(description="The chart of the page")
from extract_thinker.models.contract import Contract
from tests.models.XYCoordinate import XYCoordinate
from pydantic import Field
from typing import List, Literal


class Chart(Contract):
    classification: Literal["line", "bar", "pie"] = Field(description="The type of the chart")
    description: str = Field(description="Description of the chart")
    coordinates: List[XYCoordinate] = Field(description="The x-axis and y-axis present in the chart. Will be multiple")
    gdp_variation: str = Field(description="Description of the gdp variation")
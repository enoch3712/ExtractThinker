from extract_thinker.models.contract import Contract
from pydantic import Field


class XYCoordinate(Contract):
    x: int = Field(description="Value of the x")
    y: int = Field(description="Value of the y")
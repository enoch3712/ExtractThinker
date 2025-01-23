from pydantic import BaseModel, Field
from typing import Dict, List

class MaskContract(BaseModel):
    mapping: Dict[str, str] = Field(description="A dictionary mapping placeholders to original values")
    masked_text: str = Field(description="The masked version of the text")

    def __init__(self, masked_text: str, mapping: Dict[str, str]):
        super().__init__(masked_text=masked_text, mapping=mapping)

class MaskContractDict(BaseModel):
    mapping: Dict[str, str] = Field(description="A dictionary mapping placeholders to original values")
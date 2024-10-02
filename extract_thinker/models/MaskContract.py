from pydantic import BaseModel, Field
from typing import Dict, List

class MaskContract(BaseModel):
    masked_text: str = Field(description="The masked version of the text")
    mapping: Dict[str, str] = Field(description="A dictionary mapping placeholders to original values")
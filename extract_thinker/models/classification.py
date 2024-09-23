from typing import Any, Optional, Type
from pydantic import BaseModel, field_validator
from extract_thinker.models.contract import Contract
import os

class Classification(BaseModel):
    name: str
    description: str
    contract: Optional[Type] = None
    image: Optional[str] = None
    extractor: Optional[Any] = None

    @field_validator('contract', mode='before')
    def validate_contract(cls, v):
        if v is not None:
            if not isinstance(v, type):
                raise ValueError('contract must be a type')
            if not issubclass(v, Contract):
                raise ValueError('contract must be a subclass of Contract')
        return v

    def set_image(self, image_path: str):
        if os.path.isfile(image_path):
            self.image = image_path
        else:
            raise ValueError(f"The provided string '{image_path}' is not a valid file path.")

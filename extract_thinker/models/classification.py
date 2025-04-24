from typing import Any, Optional, Type
from pydantic import BaseModel, field_validator, Field
from uuid import UUID, uuid4
import os

class Classification(BaseModel):
    name: str
    description: str
    contract: Optional[Type] = None
    extraction_contract: Optional[Type] = None
    image: Optional[str] = None
    extractor: Optional[Any] = None
    uuid: UUID = Field(default_factory=uuid4)

    @field_validator('contract', mode='before')
    def validate_contract(cls, v):
        if v is not None:
            if not isinstance(v, type):
                raise ValueError('contract must be a type')
        return v

    @field_validator('extraction_contract', mode='before')
    def validate_extraction_contract(cls, v):
        if v is not None:
            if not isinstance(v, type):
                raise ValueError('extraction_contract must be a type')
        return v

    def set_image(self, image_path: str):
        if os.path.isfile(image_path):
            self.image = image_path
        else:
            raise ValueError(f"The provided string '{image_path}' is not a valid file path.")

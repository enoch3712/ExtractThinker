from typing import Any, Optional
from pydantic import BaseModel
from extract_thinker.models.contract import Contract


class Classification(BaseModel):
    name: str
    description: str
    contract: Optional[Contract] = None
    extractor: Optional[Any] = None

from typing import Any, Optional
from extract_thinker.models.contract import Contract
from pydantic import BaseModel


class Classification(BaseModel):
    name: str
    description: str
    contract: type[Contract]
    extractor: Optional[Any] = None

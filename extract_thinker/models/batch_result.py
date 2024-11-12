from pydantic import BaseModel
from typing import Any, List

class BatchResult(BaseModel):
	id: str
	results: List[Any]
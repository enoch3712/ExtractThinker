from typing import List, Dict
from pydantic import BaseModel

class Classification(BaseModel):
    name: str
    description: str
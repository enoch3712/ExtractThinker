from dataclasses import dataclass
from typing import List, Optional
from pydantic import BaseModel

@dataclass
class EagerDocGroup:
    pages: List[int]
    classification: str
    classification_id: Optional[int] = None

class DocGroup(BaseModel):
    pages: List[int]
    classification: str
    classification_id: Optional[int] = None

class DocGroupsEager(BaseModel):
    reasoning: str
    groupOfDocuments: List[DocGroup]
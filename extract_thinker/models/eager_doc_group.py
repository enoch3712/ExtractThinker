from dataclasses import dataclass
from typing import List
from pydantic import BaseModel

@dataclass
class EagerDocGroup:
    pages: List[str]
    classification: str

class DocGroup(BaseModel):
    pages: List[int]
    classification: str

class DocGroupsEager(BaseModel):
    reasoning: str
    groupOfDocuments: List[DocGroup]
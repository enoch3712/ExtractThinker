from dataclasses import dataclass
from typing import List

@dataclass
class EagerDocGroup:
    pages: List[str]
    classification: str


from typing import List
from pydantic import BaseModel


class DocGroup(BaseModel):
    pages: List[int]
    classification: str

class DocGroupsEager(BaseModel):
    reasoning: str
    groupOfDocuments: List[DocGroup]
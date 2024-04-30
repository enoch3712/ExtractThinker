from typing import List, Optional
from pydantic import BaseModel
from dataclasses import dataclass


class Contract(BaseModel):
    pass


class Classification(BaseModel):
    name: str
    description: str
    contract: Optional[Contract] = None


class ClassificationResponse(BaseModel):
    name: str


@dataclass
class DocGroups2:
    certainty: float
    belongs_to_same_document: bool
    classification_page1: str
    classification_page2: str


class DocGroup:
    def __init__(self):
        self.pages: List[int] = []
        self.classification: str = ""
        self.certainties: List[float] = []


class DocGroups:
    def __init__(self):
        self.doc_groups: List[DocGroup] = []

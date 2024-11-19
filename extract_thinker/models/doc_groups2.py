from typing import Optional
from pydantic import BaseModel

class DocGroups2(BaseModel):
    reasoning: Optional[str] = None
    belongs_to_same_document: bool
    classification_page1: str
    classification_page2: str
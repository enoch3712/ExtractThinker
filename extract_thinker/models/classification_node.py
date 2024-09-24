from typing import List, Optional
from pydantic import BaseModel, Field
from extract_thinker.models.classification import Classification

class ClassificationNode(BaseModel):
    name: str
    classification: Classification
    children: List['ClassificationNode'] = Field(default_factory=list)

ClassificationNode.model_rebuild()
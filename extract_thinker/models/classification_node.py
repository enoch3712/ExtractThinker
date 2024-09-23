from typing import List
from pydantic import BaseModel, Field
from extract_thinker.models.classification import Classification

class ClassificationNode(BaseModel):
    classification: Classification
    children: List['ClassificationNode'] = Field(default_factory=list)

ClassificationNode.model_rebuild()
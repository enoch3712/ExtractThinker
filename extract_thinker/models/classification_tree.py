from extract_thinker.models.classification_node import ClassificationNode
from typing import List
from pydantic import BaseModel

class ClassificationTree(BaseModel):
    nodes: List[ClassificationNode]
from typing import Optional
from pydantic import BaseModel, Field
from extract_thinker.models.classification import Classification

class ClassificationResponseInternal(BaseModel):
    confidence: int = Field(description="From 1 to 10. 10 being the highest confidence. Always integer", ge=1, le=10)
    name: str

class ClassificationResponse(ClassificationResponseInternal):
    classification: Classification
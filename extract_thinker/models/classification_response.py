from pydantic import BaseModel, Field


class ClassificationResponse(BaseModel):
    name: str
    confidence: int = Field("From 1 to 10. 10 being the highest confidence. Always integer", ge=1, le=10)
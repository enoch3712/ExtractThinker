from typing import Optional
from pydantic import BaseModel


class HallucinationResult(BaseModel):
    """Results from hallucination detection."""
    field_name: str
    hallucination_score: float  # 0.0 (definitely real) to 1.0 (definitely hallucinated)
    reasoning: Optional[str] = None
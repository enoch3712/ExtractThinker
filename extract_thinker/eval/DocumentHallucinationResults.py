from extract_thinker.eval.HallucinationResult import HallucinationResult


from pydantic import BaseModel


from typing import Dict, List


class DocumentHallucinationResults(BaseModel):
    """Hallucination detection results for an entire document."""
    doc_id: str
    overall_score: float
    field_scores: Dict[str, float]
    detailed_results: List[HallucinationResult]
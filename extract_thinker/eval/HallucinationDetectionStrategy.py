from enum import Enum, auto


class HallucinationDetectionStrategy(Enum):
    """Strategies for detecting hallucinations in extracted data."""
    LLM = auto()        # Use LLM to check if data is present in document
    HEURISTIC = auto()  # Use pattern matching and other heuristics
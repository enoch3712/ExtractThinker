from dataclasses import dataclass
from typing import List

@dataclass
class EagerDocResult:
    reason: str
    documents: List[List[int]]
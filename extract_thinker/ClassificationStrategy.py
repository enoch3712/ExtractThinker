from enum import Enum


class ClassificationStrategy(Enum):
    CONSENSUS = "consensus"
    HIGHER_ORDER = "higher_order"
    CONSENSUS_WITH_THRESHOLD = "both"
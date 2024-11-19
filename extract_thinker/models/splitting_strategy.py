from enum import Enum


class SplittingStrategy(Enum):
    EAGER = "eager"
    LAZY = "lazy"
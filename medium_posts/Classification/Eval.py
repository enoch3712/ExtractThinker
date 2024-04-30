from abc import ABC, abstractmethod

class Eval(ABC):
    def __init__(self, threshold: float):
        self.threshold = threshold

    @abstractmethod
    def evaluate(self, result):
        pass
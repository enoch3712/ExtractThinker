import math
from Classification.Eval import Eval
import numpy as np

class LogProbEval(Eval):
    def __init__(self, threshold):
        self.threshold = threshold

    def evaluate(self, result):
        # Ensure there are log probabilities to process
        if not result.choices[0].logprobs.content:
            return 0.0

        # Calculate the total log probability
        total_logprob = sum([math.exp(c.logprob) for c in result.choices[0].logprobs.content])

        # Calculate the average log probability
        avg_logprob = total_logprob / len(result.choices[0].logprobs.content)

        return avg_logprob > self.threshold
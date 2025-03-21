from typing import Dict, List, Any, Optional
import statistics
from pydantic import BaseModel

class CostMetrics:
    """
    Track and calculate token usage and cost metrics for evaluations.
    """
    
    def __init__(self):
        """Initialize the cost metrics tracker."""
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.input_tokens = []
        self.output_tokens = []
        self.total_tokens = []
        self.costs = []
        self.costs_by_doc = {}
    
    def update(self, 
               doc_id: str, 
               input_tokens: int, 
               output_tokens: int, 
               cost: float):
        """
        Update metrics with token usage and cost from a document.
        
        Args:
            doc_id: Document identifier
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens generated
            cost: Total cost in USD
        """
        self.input_tokens.append(input_tokens)
        self.output_tokens.append(output_tokens)
        self.total_tokens.append(input_tokens + output_tokens)
        self.costs.append(cost)
        self.costs_by_doc[doc_id] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": cost
        }
    
    def get_total_cost(self) -> float:
        """
        Get total cost across all documents.
        
        Returns:
            float: Total cost in USD
        """
        return sum(self.costs)
    
    def get_average_cost(self) -> float:
        """
        Get average cost per document.
        
        Returns:
            float: Average cost per document in USD
        """
        return statistics.mean(self.costs) if self.costs else 0.0
    
    def get_total_tokens(self) -> int:
        """
        Get total tokens used across all documents.
        
        Returns:
            int: Total token count
        """
        return sum(self.total_tokens)
    
    def get_average_tokens(self) -> float:
        """
        Get average tokens per document.
        
        Returns:
            float: Average token count per document
        """
        return statistics.mean(self.total_tokens) if self.total_tokens else 0.0
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all cost and token metrics.
        
        Returns:
            Dict: All metrics including totals and averages
        """
        return {
            "total_cost": self.get_total_cost(),
            "average_cost": self.get_average_cost(),
            "total_tokens": self.get_total_tokens(),
            "average_tokens": self.get_average_tokens(),
            "total_input_tokens": sum(self.input_tokens),
            "total_output_tokens": sum(self.output_tokens),
            "average_input_tokens": statistics.mean(self.input_tokens) if self.input_tokens else 0.0,
            "average_output_tokens": statistics.mean(self.output_tokens) if self.output_tokens else 0.0
        } 
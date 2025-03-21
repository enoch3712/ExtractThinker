from enum import Enum
from typing import Dict, Any, Optional, Type, Callable, Union, List
import numpy as np
from pydantic import BaseModel

class ComparisonType(str, Enum):
    """Defines different ways to compare field values."""
    EXACT = "exact"  # Perfect string/value match
    FUZZY = "fuzzy"  # Approximate string matching (Levenshtein distance)
    SEMANTIC = "semantic"  # Semantic similarity (embedding-based)
    NUMERIC = "numeric"  # Numeric comparison with tolerance
    CUSTOM = "custom"  # Custom comparison function

class FieldComparisonConfig(BaseModel):
    """Configuration for comparing different field types."""
    
    comparison_type: ComparisonType
    similarity_threshold: float = 0.8  # Default threshold for non-exact comparisons
    numeric_tolerance: float = 0.01  # Default tolerance for numeric comparisons (1%)
    custom_comparator: Optional[Callable[[Any, Any], bool]] = None
    
    def is_match(self, expected: Any, predicted: Any) -> bool:
        """
        Determine if predicted value matches expected value according to config.
        
        Args:
            expected: The expected value
            predicted: The predicted value
            
        Returns:
            bool: True if values match according to comparison criteria
        """
        if expected is None and predicted is None:
            return True
            
        if expected is None or predicted is None:
            return False
            
        if self.comparison_type == ComparisonType.EXACT:
            return self._exact_match(expected, predicted)
            
        elif self.comparison_type == ComparisonType.FUZZY:
            return self._fuzzy_match(expected, predicted)
            
        elif self.comparison_type == ComparisonType.SEMANTIC:
            return self._semantic_match(expected, predicted)
            
        elif self.comparison_type == ComparisonType.NUMERIC:
            return self._numeric_match(expected, predicted)
            
        elif self.comparison_type == ComparisonType.CUSTOM and self.custom_comparator:
            return self.custom_comparator(expected, predicted)
            
        # Default to exact match if no valid comparison type
        return self._exact_match(expected, predicted)
    
    def _exact_match(self, expected: Any, predicted: Any) -> bool:
        """Check if values match exactly."""
        # Convert both to strings for more flexible comparison
        if isinstance(expected, (list, dict)) and isinstance(predicted, (list, dict)):
            # For complex objects, compare serialized versions
            import json
            return json.dumps(expected, sort_keys=True) == json.dumps(predicted, sort_keys=True)
        return str(expected).strip() == str(predicted).strip()
    
    def _fuzzy_match(self, expected: Any, predicted: Any) -> bool:
        """Check if string values approximately match."""
        try:
            # Import Levenshtein only when needed
            from Levenshtein import ratio
            
            # Convert to strings
            expected_str = str(expected).strip()
            predicted_str = str(predicted).strip()
            
            # Calculate similarity ratio
            similarity = ratio(expected_str, predicted_str)
            return similarity >= self.similarity_threshold
        except ImportError:
            # Fallback to exact match if Levenshtein not available
            return self._exact_match(expected, predicted)
    
    def _semantic_match(self, expected: Any, predicted: Any) -> bool:
        """Check if texts are semantically similar using embeddings."""
        try:
            # Try to use an embedding model for semantic similarity
            from sentence_transformers import SentenceTransformer
            
            # Initialize the model (lazy-loaded singleton)
            if not hasattr(self.__class__, '_model'):
                self.__class__._model = SentenceTransformer('all-MiniLM-L6-v2')
                
            # Convert to strings
            expected_str = str(expected).strip()
            predicted_str = str(predicted).strip()
            
            # Calculate embeddings and cosine similarity
            embeddings = self.__class__._model.encode([expected_str, predicted_str])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            return similarity >= self.similarity_threshold
        except ImportError:
            # Try using litellm if available
            try:
                from litellm import embedding
                
                # Get embeddings
                expected_embedding = embedding(model="text-embedding-ada-002", input=str(expected).strip())
                predicted_embedding = embedding(model="text-embedding-ada-002", input=str(predicted).strip())
                
                # Calculate cosine similarity
                expected_vector = expected_embedding['data'][0]['embedding']
                predicted_vector = predicted_embedding['data'][0]['embedding']
                
                similarity = np.dot(expected_vector, predicted_vector) / (
                    np.linalg.norm(expected_vector) * np.linalg.norm(predicted_vector)
                )
                
                return similarity >= self.similarity_threshold
            except (ImportError, Exception):
                # Fallback to fuzzy match
                return self._fuzzy_match(expected, predicted)
    
    def _numeric_match(self, expected: Any, predicted: Any) -> bool:
        """Check if numeric values match within tolerance."""
        try:
            # Convert to float
            expected_num = float(expected)
            predicted_num = float(predicted)
            
            # If expected is zero, use absolute tolerance
            if expected_num == 0:
                return abs(predicted_num) <= self.numeric_tolerance
                
            # Calculate relative difference
            relative_diff = abs((predicted_num - expected_num) / expected_num)
            return relative_diff <= self.numeric_tolerance
        except (ValueError, TypeError):
            # If conversion fails, fall back to exact match
            return self._exact_match(expected, predicted)


class FieldComparisonManager:
    """
    Manages comparison configurations for different fields.
    """
    
    def __init__(self, response_model: Type[BaseModel]):
        """
        Initialize with a response model and default configurations.
        
        Args:
            response_model: The pydantic model defining the schema
        """
        self.response_model = response_model
        self.field_configs: Dict[str, FieldComparisonConfig] = {}
        
        # Set default configuration for all fields
        self._initialize_defaults()
    
    def _initialize_defaults(self):
        """Set up default comparison configurations based on field types."""
        model_fields = getattr(self.response_model, "__annotations__", {})
        
        for field_name, field_type in model_fields.items():
            # Check field type and set appropriate comparison
            if field_type in (str, int, float, bool) or field_type is str or field_type is int or field_type is float or field_type is bool:
                # For primitive types
                if field_type in (int, float) or field_type is int or field_type is float:
                    self.field_configs[field_name] = FieldComparisonConfig(comparison_type=ComparisonType.NUMERIC)
                else:
                    self.field_configs[field_name] = FieldComparisonConfig(comparison_type=ComparisonType.EXACT)
            else:
                # For complex types (lists, dicts, nested models)
                self.field_configs[field_name] = FieldComparisonConfig(comparison_type=ComparisonType.EXACT)
    
    def set_comparison(self, field_name: str, config: FieldComparisonConfig):
        """
        Set comparison configuration for a specific field.
        
        Args:
            field_name: Name of the field
            config: Comparison configuration
        """
        self.field_configs[field_name] = config
    
    def get_comparison(self, field_name: str) -> FieldComparisonConfig:
        """
        Get comparison configuration for a specific field.
        
        Args:
            field_name: Name of the field
            
        Returns:
            FieldComparisonConfig: Configuration for the field
        """
        return self.field_configs.get(field_name, FieldComparisonConfig(comparison_type=ComparisonType.EXACT))
    
    def compare_values(self, field_name: str, expected: Any, predicted: Any) -> bool:
        """
        Compare values using the appropriate comparison for the given field.
        
        Args:
            field_name: Name of the field
            expected: Expected value
            predicted: Predicted value
            
        Returns:
            bool: True if values match according to field's comparison criteria
        """
        config = self.get_comparison(field_name)
        return config.is_match(expected, predicted) 
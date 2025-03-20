from typing import Dict, Any, Optional, List, Union
import re
from pydantic import BaseModel
from extract_thinker.eval.DocumentHallucinationResults import DocumentHallucinationResults
from extract_thinker.eval.HallucinationDetectionStrategy import HallucinationDetectionStrategy
from extract_thinker.eval.HallucinationResult import HallucinationResult
from extract_thinker.llm import LLM

class HallucinationCheckResponse(BaseModel):
    """Response model for LLM hallucination check."""
    is_contradicted: bool  # Whether the field contradicts the document context
    score: float  # 0.0 (definitely in document) to 1.0 (definitely hallucinated)
    reasoning: str  # Explanation for the assessment

class HallucinationDetector:
    """
    Detects potential hallucinations in extracted data by comparing to source document.
    
    Following the Confident AI approach, hallucination is calculated as:
    Hallucination = Number of Contradicted Fields / Total Number of Fields
    """
    
    def __init__(
        self, 
        llm: Optional[LLM] = None, 
        threshold: float = 0.7,
        strategy: HallucinationDetectionStrategy = None
    ):
        """
        Initialize the hallucination detector.
        
        Args:
            llm: Optional LLM instance to use for hallucination detection (required for LLM strategy)
            threshold: Score threshold above which a field is considered hallucinated
            strategy: The strategy to use for hallucination detection. If None, will use
                      LLM if an LLM is provided, otherwise will use HEURISTIC.
        """
        self.llm = llm
        self.threshold = threshold
        
        # Set default strategy based on LLM availability if not specified
        if strategy is None:
            self.strategy = HallucinationDetectionStrategy.LLM if llm else HallucinationDetectionStrategy.HEURISTIC
        else:
            self.strategy = strategy
            
        # Validate that we have an LLM if using LLM strategy
        if self.strategy == HallucinationDetectionStrategy.LLM and not self.llm:
            raise ValueError("LLM strategy requires an LLM instance to be provided")
        
    def detect_hallucinations(
        self, 
        extracted_data: Dict[str, Any],
        document_text: str
    ) -> DocumentHallucinationResults:
        """
        Detect potential hallucinations in extracted fields.
        
        Args:
            extracted_data: The extracted structured data
            document_text: The raw text from the document
            
        Returns:
            DocumentHallucinationResults with field-level hallucination scores
        """
        detailed_results = []
        field_scores = {}
        
        contradicted_count = 0
        total_fields = 0
        
        # Process each field in the extracted data
        for field_name, field_value in extracted_data.items():
            if self._should_skip_field(field_name, field_value):
                continue
                
            total_fields += 1
            
            # Use different detection strategies based on field value type
            result = self._detect_field_hallucination(field_name, field_value, document_text)
            detailed_results.append(result)
            field_scores[field_name] = result.hallucination_score
            
            # Count as contradicted if score exceeds threshold
            if result.hallucination_score >= self.threshold:
                contradicted_count += 1
        
        # Calculate overall hallucination score as per Confident AI approach
        # Hallucination = Number of Contradicted Fields / Total Number of Fields
        overall_score = contradicted_count / total_fields if total_fields > 0 else 0.0
        
        return DocumentHallucinationResults(
            doc_id=extracted_data.get("doc_id", "unknown"),
            overall_score=overall_score,
            field_scores=field_scores,
            detailed_results=detailed_results
        )
    
    def _should_skip_field(self, field_name: str, field_value: Any) -> bool:
        """Determine if a field should be skipped in hallucination detection."""
        # Skip None values
        if field_value is None:
            return True
            
        # Skip metadata fields
        if field_name in ("doc_id", "metadata", "confidence"):
            return True
            
        return False
    
    def _detect_field_hallucination(
        self, 
        field_name: str, 
        field_value: Any, 
        document_text: str
    ) -> HallucinationResult:
        """
        Detect hallucination for a specific field.
        
        Based on the field value type and selected strategy, determines if the field
        contradicts or is unsupported by the document context.
        """
        if isinstance(field_value, (str, int, float, bool)):
            if self.strategy == HallucinationDetectionStrategy.LLM:
                return self._llm_hallucination_check(field_name, field_value, document_text)
            elif self.strategy == HallucinationDetectionStrategy.HEURISTIC:
                return self._heuristic_hallucination_check(field_name, field_value, document_text)
            else:
                raise ValueError(f"Unknown hallucination detection strategy: {self.strategy}")
        
        elif isinstance(field_value, list):
            return self._list_hallucination_check(field_name, field_value, document_text)
        elif isinstance(field_value, dict):
            return self._dict_hallucination_check(field_name, field_value, document_text)
        
        return HallucinationResult(
            field_name=field_name,
            hallucination_score=0.5,
            reasoning="Unhandled field type"
        )
    
    def _heuristic_hallucination_check(
        self, 
        field_name: str, 
        field_value: Any, 
        document_text: str
    ) -> HallucinationResult:
        """
        Use heuristic methods to check for hallucinations.
        
        Following the Confident AI approach, we determine if the field contradicts
        or is unsupported by the document context.
        """
        field_str = str(field_value).strip().lower()
        doc_text_lower = document_text.lower()
        
        # Direct string matching
        if field_str in doc_text_lower:
            return HallucinationResult(
                field_name=field_name,
                hallucination_score=0.0,
                reasoning="Value found directly in document text"
            )
        
        # Word-level matching
        words = re.findall(r'\b\w+\b', field_str)
        if words and all(word.lower() in doc_text_lower for word in words):
            return HallucinationResult(
                field_name=field_name,
                hallucination_score=0.3,
                reasoning="All words found in document, but not the exact phrase"
            )
        
        # Partial matching
        if len(field_str) > 3 and any(field_str[i:i+4] in doc_text_lower for i in range(len(field_str)-3)):
            return HallucinationResult(
                field_name=field_name,
                hallucination_score=0.6,
                reasoning="Partial match found in document"
            )
            
        # No match - considered contradicted/hallucinated
        return HallucinationResult(
            field_name=field_name,
            hallucination_score=0.9,
            reasoning="No significant match found in document - considered hallucinated"
        )
    
    def _llm_hallucination_check(
        self, 
        field_name: str, 
        field_value: Any, 
        document_text: str
    ) -> HallucinationResult:
        """
        Use LLM to check for hallucinations following the Confident AI approach.
        
        We ask the LLM to determine if the extracted field contradicts the document context.
        """
        if not self.llm:
            raise ValueError("LLM strategy selected but no LLM instance provided")
            
        prompt = f"""
        I need to determine if an extracted field might be hallucinated (not present in or contradicted by the source document).
        
        Field name: {field_name}
        Extracted value: {field_value}
        
        Document text (truncated):
        ---
        {document_text[:2000]}
        ---
        
        Task: Determine if this extracted value contradicts the document or isn't supported by it.
        
        Follow these guidelines:
        1. If the value is directly stated or can be reasonably inferred from the document: NOT HALLUCINATED
        2. If the value contradicts information in the document: HALLUCINATED
        3. If the value is completely unrelated or unsupported by the document: HALLUCINATED
        
        Respond with:
        1. Whether the field is contradicted or not (true/false)
        2. A hallucination score from 0.0 (definitely in document) to 1.0 (definitely hallucinated)
        3. Brief reasoning for your assessment
        """
        
        try:
            # Use the LLM's request method with our response model
            response = self.llm.request(
                messages=[{"role": "user", "content": prompt}],
                response_model=HallucinationCheckResponse
            )
            
            return HallucinationResult(
                field_name=field_name,
                hallucination_score=response.score,
                reasoning=response.reasoning
            )
        except Exception as e:
            # Fallback to heuristic if LLM fails
            result = self._heuristic_hallucination_check(field_name, field_value, document_text)
            result.reasoning = f"LLM error: {str(e)}. {result.reasoning}"
            return result
    
    def _list_hallucination_check(
        self, 
        field_name: str, 
        field_value: List[Any], 
        document_text: str
    ) -> HallucinationResult:
        """Check if a list field is hallucinated."""
        if not field_value:
            return HallucinationResult(
                field_name=field_name,
                hallucination_score=0.0,
                reasoning="Empty list"
            )
        
        # For simple lists, check each value
        if all(isinstance(item, (str, int, float, bool)) for item in field_value):
            item_scores = []
            for item in field_value:
                result = self._detect_field_hallucination(f"{field_name}[item]", item, document_text)
                item_scores.append(result.hallucination_score)
            
            avg_score = sum(item_scores) / len(item_scores)
            return HallucinationResult(
                field_name=field_name,
                hallucination_score=avg_score,
                reasoning=f"Average hallucination score of {len(item_scores)} list items"
            )
        
        # For complex lists (lists of objects), use LLM or simplified check
        if self.strategy == HallucinationDetectionStrategy.LLM:
            return self._llm_hallucination_check(field_name, field_value, document_text)
        else:
            return HallucinationResult(
                field_name=field_name,
                hallucination_score=0.5,
                reasoning="Complex list field - limited heuristic check"
            )
    
    def _dict_hallucination_check(
        self, 
        field_name: str, 
        field_value: Dict[str, Any], 
        document_text: str
    ) -> HallucinationResult:
        """Check if a dictionary field is hallucinated."""
        # For nested dicts, check key values
        subfield_results = []
        for key, value in field_value.items():
            subfield_name = f"{field_name}.{key}"
            result = self._detect_field_hallucination(subfield_name, value, document_text)
            subfield_results.append(result)
        
        # Calculate average score
        if subfield_results:
            avg_score = sum(r.hallucination_score for r in subfield_results) / len(subfield_results)
            return HallucinationResult(
                field_name=field_name,
                hallucination_score=avg_score,
                reasoning=f"Average hallucination score of {len(subfield_results)} subfields"
            )
        
        return HallucinationResult(
            field_name=field_name,
            hallucination_score=0.5,
            reasoning="Empty dictionary field"
        ) 
from typing import Dict, Any, Optional, List, Union
import re
from pydantic import BaseModel

class HallucinationResult(BaseModel):
    """Results from hallucination detection."""
    field_name: str
    hallucination_score: float  # 0.0 (definitely real) to 1.0 (definitely hallucinated)
    reasoning: Optional[str] = None
    
class DocumentHallucinationResults(BaseModel):
    """Hallucination detection results for an entire document."""
    doc_id: str
    overall_score: float
    field_scores: Dict[str, float]
    detailed_results: List[HallucinationResult]
    
class HallucinationDetector:
    """
    Detects potential hallucinations in extracted data by comparing to source document.
    """
    
    def __init__(self, llm=None, threshold: float = 0.7):
        """
        Initialize the hallucination detector.
        
        Args:
            llm: Optional LLM to use for hallucination detection (recommended)
            threshold: Score threshold above which a field is considered hallucinated
        """
        self.llm = llm
        self.threshold = threshold
        
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
        
        # Process each field in the extracted data
        for field_name, field_value in extracted_data.items():
            if self._should_skip_field(field_name, field_value):
                continue
                
            # Use different detection strategies based on field value type
            result = self._detect_field_hallucination(field_name, field_value, document_text)
            detailed_results.append(result)
            field_scores[field_name] = result.hallucination_score
        
        # Calculate overall hallucination score
        overall_score = sum(field_scores.values()) / len(field_scores) if field_scores else 0.0
        
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
        """Detect hallucination for a specific field."""
        # For simple text fields, do basic text matching
        if isinstance(field_value, (str, int, float, bool)):
            if self.llm:
                # Use LLM-based detection
                return self._llm_hallucination_check(field_name, field_value, document_text)
            else:
                # Use heuristic detection
                return self._heuristic_hallucination_check(field_name, field_value, document_text)
        
        # For complex fields (lists, dicts), handle accordingly
        elif isinstance(field_value, list):
            return self._list_hallucination_check(field_name, field_value, document_text)
        elif isinstance(field_value, dict):
            return self._dict_hallucination_check(field_name, field_value, document_text)
        
        # Default case for unhandled types
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
        """Use heuristic methods to check for hallucinations."""
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
            
        # No match
        return HallucinationResult(
            field_name=field_name,
            hallucination_score=0.9,
            reasoning="No significant match found in document"
        )
    
    def _llm_hallucination_check(
        self, 
        field_name: str, 
        field_value: Any, 
        document_text: str
    ) -> HallucinationResult:
        """Use LLM to check for hallucinations."""
        prompt = f"""
        I need to determine if an extracted field might be hallucinated (not actually present in the source document).
        
        Field name: {field_name}
        Extracted value: {field_value}
        
        Document text (truncated):
        ---
        {document_text[:2000]}
        ---
        
        Is this extracted value present in the document or can it be reasonably inferred?
        
        Respond with:
        1. A hallucination score from 0.0 (definitely in document) to 1.0 (definitely hallucinated)
        2. Brief reasoning for your assessment
        
        Format: 
        Score: [0.0-1.0]
        Reasoning: [your explanation]
        """
        
        try:
            response = self.llm.raw_completion(
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract score and reasoning
            score_match = re.search(r'Score:\s*(0\.\d+|1\.0)', response)
            reasoning_match = re.search(r'Reasoning:\s*(.*?)(?:\n|$)', response, re.DOTALL)
            
            score = float(score_match.group(1)) if score_match else 0.5
            reasoning = reasoning_match.group(1).strip() if reasoning_match else None
            
            return HallucinationResult(
                field_name=field_name,
                hallucination_score=score,
                reasoning=reasoning
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
        if self.llm:
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
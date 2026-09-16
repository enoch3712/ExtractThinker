"""Numeric classification responses used internally by document splitters."""
from typing import List, Optional
from pydantic import BaseModel, Field, StrictInt


class NumericPagePair(BaseModel):
    reasoning: Optional[str] = None
    belongs_to_same_document: bool
    classification_page1: StrictInt = Field(ge=1)
    classification_page2: StrictInt = Field(ge=1)


class NumericDocumentGroup(BaseModel):
    pages: List[StrictInt]
    classification: StrictInt = Field(ge=1)


class NumericDocumentGroups(BaseModel):
    reasoning: str
    groupOfDocuments: List[NumericDocumentGroup]


def resolve_classification(classifications, identifier=None, name=None):
    """Resolve a request-local numeric ID, or an unambiguous legacy name."""
    if identifier is not None:
        if isinstance(identifier, bool) or not isinstance(identifier, int) or not 1 <= identifier <= len(classifications):
            raise ValueError(f"Unknown classification ID: {identifier}")
        return classifications[identifier - 1]
    matches = [item for item in classifications if item.name == name]
    if len(matches) != 1:
        raise ValueError(f"Classification name is unknown or ambiguous: {name!r}; use classification_id")
    return matches[0]

"""Reusable, provider-independent types for document coordinates and signatures."""
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class BoundingBox(BaseModel):
    """Normalized rectangle on a rendered page, with a top-left origin."""

    page: int = Field(ge=1, description="One-based source page number")
    x0: float = Field(ge=0, le=1, description="Left edge as a fraction of page width")
    y0: float = Field(ge=0, le=1, description="Top edge as a fraction of page height")
    x1: float = Field(ge=0, le=1, description="Right edge as a fraction of page width")
    y1: float = Field(ge=0, le=1, description="Bottom edge as a fraction of page height")

    @model_validator(mode="after")
    def validate_edges(self):
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("Bounding box right/bottom edges must not precede left/top edges")
        return self


class DocumentRegion(BaseModel):
    """Source text with coordinates supplied by a document loader."""

    text: str
    bounding_box: BoundingBox
    confidence: Optional[float] = Field(default=None, ge=0, le=1)


class Signature(BaseModel):
    """A detected signature mark, not cryptographic or identity verification."""

    present: bool = Field(description="Whether a signature mark is present")
    signer: Optional[str] = Field(default=None, description="Signer name if known from the source")
    bounding_box: Optional[BoundingBox] = None
    confidence: Optional[float] = Field(default=None, ge=0, le=1)

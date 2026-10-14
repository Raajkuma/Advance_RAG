"""Shared data structures; metadata is preserved through every stage."""
from dataclasses import dataclass, asdict
from typing import Any
from pydantic import BaseModel, Field
from typing import Optional
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class Document(BaseModel):
    title: str = ""
    text: str
    source: str
    page: Optional[int | str] = 1
    source_type: Optional[str] = "document"
    extraction_method: Optional[str] = "standard"  # Added to prevent missing attribute error
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)  # Added to prevent missing attribute error
                                               
@dataclass
class SourceDocument:
    text: str
    source: str
    source_type: str
    page: int | None = None
    title: str = ""
    language: str = "unknown"
    extraction_method: str = "text"
    metadata: dict[str, Any] | None = None

    def to_dict(self):
        return asdict(self)

@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    source_type: str
    page: int | None = None
    title: str = ""
    language: str = "unknown"
    extraction_method: str = "text"
    metadata: dict[str, Any] | None = None

    def to_dict(self):
        return asdict(self)

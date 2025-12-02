from pydantic import BaseModel
from typing import List, Dict, Any


class FileSpec(BaseModel):
    path: str
    content: str


class CycleSpec(BaseModel):
    id: str
    graph: Dict[str, Any]
    files: List[FileSpec]
    metadata: Dict[str, Any] = {}


class CycleDescription(BaseModel):
    text: str
    highlights: List[Dict[str, Any]] = []


class Patch(BaseModel):
    path: str
    original: str
    patched: str


class RefactorProposal(BaseModel):
    patches: List[Patch]
    rationale: str
    metadata: Dict[str, Any] = {}

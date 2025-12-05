"""Pydantic schemas for cycle-refactoring pipeline data contracts.

These schemas are used at entry points to validate input and provide
type-safe data structures throughout the pipeline. Agents receive
model instances rather than raw dicts.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional


# =============================================================================
# Input Schemas
# =============================================================================


class FileSpec(BaseModel):
    """A source file with its path and content."""
    path: str
    content: str = ""


class GraphSpec(BaseModel):
    """Dependency graph with nodes and edges."""
    nodes: List[str] = Field(default_factory=list)
    edges: List[List[str]] = Field(default_factory=list)


class CycleSpec(BaseModel):
    """Canonical cycle specification - the primary input to the pipeline."""
    id: str
    graph: GraphSpec
    files: List[FileSpec]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("files", mode="before")
    @classmethod
    def convert_file_dicts(cls, v):
        """Allow files to be passed as dicts and convert to FileSpec."""
        if isinstance(v, list):
            return [FileSpec(**f) if isinstance(f, dict) else f for f in v]
        return v

    @field_validator("graph", mode="before")
    @classmethod
    def convert_graph_dict(cls, v):
        """Allow graph to be passed as dict and convert to GraphSpec."""
        if isinstance(v, dict):
            return GraphSpec(**v)
        return v

    def get_file_paths(self) -> List[str]:
        """Return list of file paths in this cycle."""
        return [f.path for f in self.files]

    def get_file_content(self, path: str) -> Optional[str]:
        """Get content of a file by path."""
        for f in self.files:
            if f.path == path or f.path.endswith(path):
                return f.content
        return None


# =============================================================================
# Agent Output Schemas
# =============================================================================


class CycleDescription(BaseModel):
    """Output from DescriberAgent - describes the cycle in natural language."""
    text: str
    highlights: List[Dict[str, Any]] = Field(default_factory=list)


class Patch(BaseModel):
    """A single file patch with before/after content."""
    path: str
    original: str = ""
    patched: str = ""
    diff: str = ""


class RefactorProposal(BaseModel):
    """Output from RefactorAgent - proposed patches to break the cycle."""
    patches: List[Patch] = Field(default_factory=list)
    rationale: str = ""
    llm_response: Optional[str] = None

    @field_validator("patches", mode="before")
    @classmethod
    def convert_patch_dicts(cls, v):
        """Allow patches to be passed as dicts and convert to Patch."""
        if isinstance(v, list):
            return [Patch(**p) if isinstance(p, dict) else p for p in v]
        return v


class ValidationIssue(BaseModel):
    """A single validation issue found in a proposal."""
    path: str = ""
    line: Optional[int] = None
    comment: str = ""
    severity: str = "major"  # critical, major, minor, info
    issue_type: str = "semantic"  # syntax, semantic, cycle


class ValidationReport(BaseModel):
    """Output from ValidatorAgent - approval decision and feedback."""
    approved: bool = False
    decision: str = "NEEDS_REVISION"  # "APPROVED" | "NEEDS_REVISION"
    summary: str = ""
    issues: List[ValidationIssue] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)

    @field_validator("issues", mode="before")
    @classmethod
    def convert_issue_dicts(cls, v):
        """Allow issues to be passed as dicts and convert to ValidationIssue."""
        if isinstance(v, list):
            return [ValidationIssue(**i) if isinstance(i, dict) else i for i in v]
        return v


class Explanation(BaseModel):
    """Output from ExplainerAgent - human-readable summary of the refactor."""
    title: str = "Refactor Explanation"
    summary: str = ""
    explanation: str = ""
    impact: List[str] = Field(default_factory=list)
    followup: List[str] = Field(default_factory=list)

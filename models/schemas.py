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


class NodeFileMapping(BaseModel):
    """Explicit mapping between a graph node and its source file(s)."""
    node: str  # Node name in the graph (e.g., "SensorStore", "ISensorStore")
    file_path: str  # Full path to the file containing this node
    symbol: Optional[str] = None  # Specific symbol within the file (e.g., class name)
    line_start: Optional[int] = None  # Starting line of the symbol
    line_end: Optional[int] = None  # Ending line of the symbol


class CycleSpec(BaseModel):
    """Canonical cycle specification - the primary input to the pipeline."""
    id: str
    graph: GraphSpec
    files: List[FileSpec]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    node_file_map: List[NodeFileMapping] = Field(default_factory=list)

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

    @field_validator("node_file_map", mode="before")
    @classmethod
    def convert_node_file_map(cls, v):
        """Allow node_file_map to be passed as dicts and convert to NodeFileMapping."""
        if isinstance(v, list):
            return [NodeFileMapping(**m) if isinstance(m, dict) else m for m in v]
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

    def get_file_for_node(self, node: str) -> Optional[str]:
        """Get file path for a graph node using explicit mapping or heuristics.
        
        Args:
            node: Node name from the graph
            
        Returns:
            File path if found, None otherwise
        """
        # First try explicit mapping
        for mapping in self.node_file_map:
            if mapping.node == node:
                return mapping.file_path
        
        # Fallback: try to match node name to file paths
        node_lower = node.lower()
        node_clean = node_lower.replace(".", "").replace("_", "")
        
        for f in self.files:
            path = f.path
            basename = path.split("/")[-1].split("\\")[-1]
            name_no_ext = basename.rsplit(".", 1)[0].lower()
            name_clean = name_no_ext.replace(".", "").replace("_", "")
            
            # Direct match
            if node_lower == name_no_ext:
                return path
            # Clean match (ignoring underscores/dots)
            if node_clean == name_clean:
                return path
            # Node contained in filename or vice versa
            if node_clean in name_clean or name_clean in node_clean:
                return path
        
        return None

    def get_node_for_file(self, file_path: str) -> Optional[str]:
        """Get graph node for a file path using explicit mapping or heuristics.
        
        Args:
            file_path: Path to a file
            
        Returns:
            Node name if found, None otherwise
        """
        # First try explicit mapping
        for mapping in self.node_file_map:
            if mapping.file_path == file_path or file_path.endswith(mapping.file_path):
                return mapping.node
        
        # Fallback: match file to node
        basename = file_path.split("/")[-1].split("\\")[-1]
        name_no_ext = basename.rsplit(".", 1)[0].lower()
        
        for node in self.graph.nodes:
            node_lower = node.lower()
            if node_lower == name_no_ext:
                return node
            if node_lower in name_no_ext or name_no_ext in node_lower:
                return node
        
        return None

    def build_node_file_map_auto(self) -> List[NodeFileMapping]:
        """Auto-generate node-to-file mappings based on heuristics.
        
        Returns:
            List of NodeFileMapping for all matched nodes
        """
        mappings = []
        
        for node in self.graph.nodes:
            file_path = self.get_file_for_node(node)
            if file_path:
                mappings.append(NodeFileMapping(
                    node=node,
                    file_path=file_path,
                ))
        
        return mappings


# =============================================================================
# Agent Output Schemas
# =============================================================================


class CycleDescription(BaseModel):
    """Output from DescriberAgent - describes the cycle in natural language."""
    text: str
    highlights: List[Dict[str, Any]] = Field(default_factory=list)


class Patch(BaseModel):
    """A single file patch with before/after content and metadata."""
    path: str
    original: str = ""
    patched: str = ""
    diff: str = ""
    # Patch metadata for tracking what happened
    status: str = "applied"  # applied, reverted, partial, unchanged, failed
    warnings: List[str] = Field(default_factory=list)
    confidence: float = 1.0  # 0.0-1.0 confidence in the patch
    applied_blocks: int = 0  # For SEARCH/REPLACE: how many blocks applied
    total_blocks: int = 0    # For SEARCH/REPLACE: total blocks attempted
    revert_reason: str = ""  # If reverted, why
    pre_validated: bool = False  # True if RefactorAgent already validated this patch
    validation_issues: List[str] = Field(default_factory=list)  # Issues found during pre-validation


class RevertedFile(BaseModel):
    """A file that was reverted due to validation issues."""
    path: str
    reason: str
    warnings: List[str] = Field(default_factory=list)
    original_patched: Optional[str] = None  # What the patched content was before revert


class RefactorProposal(BaseModel):
    """Output from RefactorAgent - proposed patches to break the cycle."""
    patches: List[Patch] = Field(default_factory=list)
    reverted_files: List[RevertedFile] = Field(default_factory=list)
    rationale: str = ""
    llm_response: Optional[str] = None

    @field_validator("patches", mode="before")
    @classmethod
    def convert_patch_dicts(cls, v):
        """Allow patches to be passed as dicts and convert to Patch."""
        if isinstance(v, list):
            return [Patch(**p) if isinstance(p, dict) else p for p in v]
        return v

    @field_validator("reverted_files", mode="before")
    @classmethod
    def convert_reverted_dicts(cls, v):
        """Allow reverted_files to be passed as dicts and convert to RevertedFile."""
        if isinstance(v, list):
            return [RevertedFile(**rf) if isinstance(rf, dict) else rf for rf in v]
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
    # Enriched retry context (added by orchestrator for retry loop)
    previous_reverted_files: List[Dict[str, Any]] = Field(default_factory=list)
    previous_attempt_summary: str = ""
    failed_strategies: List[str] = Field(default_factory=list)
    iteration: int = 1
    remaining_attempts: int = 0

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


# =============================================================================
# RefactorRoadmap - Demo-friendly output showing progress even on failures
# =============================================================================


class FailureClassification(BaseModel):
    """Classification of why a patch or attempt failed."""
    category: str = "unknown"  # hallucination, syntax, no_op, search_mismatch, validation, compile
    description: str = ""
    evidence: List[str] = Field(default_factory=list)  # Specific evidence for the classification
    recoverable: bool = True  # Could this be fixed with another attempt?


class PartialAttempt(BaseModel):
    """A refactoring attempt that was partially completed or failed."""
    file_path: str
    intended_changes: List[str] = Field(default_factory=list)  # What we tried to do
    actual_changes: List[str] = Field(default_factory=list)    # What we managed to do
    failure_reason: str = ""
    failure_classification: Optional[FailureClassification] = None
    raw_llm_output: Optional[str] = None  # For debugging
    suggested_fix: str = ""  # What a human should do to complete this


class ScaffoldFile(BaseModel):
    """A new file created during scaffolding phase (interfaces, abstractions)."""
    path: str
    content: str
    purpose: str = ""  # "interface", "abstract_class", "shared_module"
    validated: bool = False  # True if passed syntax/compile check
    validation_errors: List[str] = Field(default_factory=list)


class RefactorRoadmap(BaseModel):
    """Demo-friendly output showing refactoring progress, even on partial failures.
    
    This provides visibility into what was attempted, what succeeded, and what
    remains to be done - useful for demonstrations and iterative refinement.
    """
    cycle_id: str = ""
    strategy_chosen: str = ""  # interface_extraction, dependency_inversion, etc.
    strategy_rationale: str = ""
    confidence: float = 0.0  # 0-1 overall confidence in the approach
    
    # Scaffolding phase results (for interface extraction)
    scaffold_files: List[ScaffoldFile] = Field(default_factory=list)
    scaffold_success: bool = True
    
    # What we successfully generated
    successful_patches: List[Patch] = Field(default_factory=list)
    
    # What we attempted but failed
    partial_attempts: List[PartialAttempt] = Field(default_factory=list)
    
    # Human-actionable next steps
    remaining_work: List[str] = Field(default_factory=list)
    
    # Minimal diff recommendation (if enabled)
    minimal_diff_target: Optional[str] = None  # The single edge to break
    minimal_diff_patch: Optional[Patch] = None  # The minimal patch if generated
    
    # For demo: formatted executive summary
    executive_summary: str = ""
    
    # Raw data for debugging
    llm_responses: List[str] = Field(default_factory=list)
    total_llm_calls: int = 0
    total_tokens_used: int = 0
    
    @field_validator("successful_patches", mode="before")
    @classmethod
    def convert_patch_dicts(cls, v):
        if isinstance(v, list):
            return [Patch(**p) if isinstance(p, dict) else p for p in v]
        return v
    
    @field_validator("scaffold_files", mode="before")
    @classmethod
    def convert_scaffold_dicts(cls, v):
        if isinstance(v, list):
            return [ScaffoldFile(**s) if isinstance(s, dict) else s for s in v]
        return v
    
    @field_validator("partial_attempts", mode="before")
    @classmethod
    def convert_attempt_dicts(cls, v):
        if isinstance(v, list):
            return [PartialAttempt(**a) if isinstance(a, dict) else a for a in v]
        return v
    
    def generate_executive_summary(self) -> str:
        """Generate a human-readable summary for demos."""
        lines = [f"## Refactoring Roadmap: {self.cycle_id}", ""]
        
        # Strategy
        lines.append(f"**Strategy**: {self.strategy_chosen}")
        if self.strategy_rationale:
            lines.append(f"*{self.strategy_rationale}*")
        lines.append(f"**Confidence**: {self.confidence:.0%}")
        lines.append("")
        
        # Scaffolding
        if self.scaffold_files:
            lines.append("### Scaffolding (New Files Created)")
            for sf in self.scaffold_files:
                status = "✅" if sf.validated else "❌"
                lines.append(f"- {status} `{sf.path}` - {sf.purpose}")
            lines.append("")
        
        # Successes
        if self.successful_patches:
            lines.append("### Successful Changes")
            for patch in self.successful_patches:
                lines.append(f"- ✅ `{patch.path}` - {patch.status}")
            lines.append("")
        
        # Partial attempts
        if self.partial_attempts:
            lines.append("### Partial/Failed Attempts")
            for attempt in self.partial_attempts:
                category = attempt.failure_classification.category if attempt.failure_classification else "unknown"
                lines.append(f"- ⚠️ `{attempt.file_path}` - {category}: {attempt.failure_reason[:100]}")
            lines.append("")
        
        # Remaining work
        if self.remaining_work:
            lines.append("### Remaining Work (Human Action Required)")
            for i, work in enumerate(self.remaining_work, 1):
                lines.append(f"{i}. {work}")
            lines.append("")
        
        # Minimal diff
        if self.minimal_diff_target:
            lines.append("### Minimal Diff Recommendation")
            lines.append(f"Focus on breaking the edge: **{self.minimal_diff_target}**")
            if self.minimal_diff_patch:
                lines.append(f"Suggested file: `{self.minimal_diff_patch.path}`")
        
        return "\n".join(lines)


# =============================================================================
# Dependency Analysis Schemas (Optional Pre-Pipeline Phase)
# =============================================================================


class ImportInfo(BaseModel):
    """A single import statement extracted from source code."""
    source_file: str  # File containing the import
    imported_from: str  # The import path as written in code
    resolved_path: Optional[str] = None  # Resolved absolute path (if resolvable)
    import_type: str = "es6"  # "es6", "commonjs", "dynamic", "type-only"
    symbols: List[str] = Field(default_factory=list)  # What's imported
    is_external: bool = False  # True if from node_modules


class DependencyGraph(BaseModel):
    """A project's complete dependency graph."""
    root_dir: str  # Project root directory
    nodes: List[str] = Field(default_factory=list)  # File paths (relative to root)
    edges: List[List[str]] = Field(default_factory=list)  # [from, to] pairs
    imports: List[ImportInfo] = Field(default_factory=list)  # Detailed import info
    metadata: Dict[str, Any] = Field(default_factory=dict)  # tsconfig, package.json info
    
    def to_graph_spec(self) -> GraphSpec:
        """Convert to GraphSpec format used by existing pipeline."""
        return GraphSpec(nodes=self.nodes, edges=self.edges)
    
    def get_internal_edges(self) -> List[List[str]]:
        """Get only internal (non-node_modules) edges."""
        external_imports = {i.imported_from for i in self.imports if i.is_external}
        return [e for e in self.edges if e[1] not in external_imports]

    @field_validator("imports", mode="before")
    @classmethod
    def convert_import_dicts(cls, v):
        if isinstance(v, list):
            return [ImportInfo(**i) if isinstance(i, dict) else i for i in v]
        return v


class DetectedCycle(BaseModel):
    """A cycle detected in the dependency graph."""
    id: str  # Unique identifier for this cycle
    nodes: List[str] = Field(default_factory=list)  # Files in the cycle
    edges: List[List[str]] = Field(default_factory=list)  # Edges forming the cycle
    severity: str = "major"  # "critical", "major", "minor"
    cycle_type: str = "direct"  # "direct", "indirect", "type-only"
    description: str = ""  # Human-readable description
    # Impact scoring (enhanced cycle detection)
    impact_score: Optional[float] = None  # Weighted importance score
    impact_explanation: Optional[str] = None  # Why this cycle is important
    layers_involved: List[str] = Field(default_factory=list)  # Architectural layers in cycle
    
    def to_cycle_spec(self, base_dir: str, file_contents: Optional[Dict[str, str]] = None) -> "CycleSpec":
        """Convert to CycleSpec for the existing refactoring pipeline.
        
        Args:
            base_dir: Base directory for resolving file paths
            file_contents: Optional dict mapping file paths to their contents.
                          If not provided, will attempt to read files from disk.
            
        Returns:
            CycleSpec ready for the refactoring pipeline
        """
        from pathlib import Path
        
        files = []
        for node in self.nodes:
            # Get content from provided dict or read from disk
            if file_contents and node in file_contents:
                content = file_contents[node]
            else:
                # Try to read from disk
                try:
                    file_path = Path(base_dir) / node if base_dir else Path(node)
                    if file_path.exists():
                        content = file_path.read_text(encoding="utf-8", errors="ignore")
                    else:
                        content = ""
                except Exception:
                    content = ""
            
            files.append(FileSpec(path=node, content=content))
        
        # Build node-file mapping (for TS/JS, node name = file path)
        node_file_map = [
            NodeFileMapping(node=node, file_path=node)
            for node in self.nodes
        ]
        
        return CycleSpec(
            id=self.id,
            graph=GraphSpec(nodes=self.nodes, edges=self.edges),
            files=files,
            metadata={
                "severity": self.severity,
                "cycle_type": self.cycle_type,
                "auto_detected": True,
            },
            node_file_map=node_file_map,
        )


class AnalysisResult(BaseModel):
    """Result of analyzing a project for cyclic dependencies."""
    project_dir: str
    graph: Optional[DependencyGraph] = None
    cycles: List[DetectedCycle] = Field(default_factory=list)
    total_files: int = 0
    total_imports: int = 0
    analysis_time_seconds: float = 0.0
    tool_used: str = ""
    errors: List[str] = Field(default_factory=list)
    
    @property
    def has_cycles(self) -> bool:
        return len(self.cycles) > 0
    
    @property
    def cycle_count(self) -> int:
        return len(self.cycles)

    @field_validator("cycles", mode="before")
    @classmethod
    def convert_cycle_dicts(cls, v):
        if isinstance(v, list):
            return [DetectedCycle(**c) if isinstance(c, dict) else c for c in v]
        return v

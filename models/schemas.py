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


# =============================================================================
# Suggestion Mode - Human-reviewable refactoring suggestions
# =============================================================================


class CodeChange(BaseModel):
    """A single code change within a suggestion."""
    line_start: int = 0
    line_end: int = 0
    original_code: str = ""
    suggested_code: str = ""
    change_type: str = "modify"  # "modify", "add", "remove"
    
    # Additional context for operators
    why_needed: str = ""  # Specific reason this change breaks the cycle
    potential_issues: List[str] = Field(default_factory=list)  # What could go wrong
    dependencies: List[str] = Field(default_factory=list)  # Other changes this depends on


class RefactorSuggestion(BaseModel):
    """A single refactoring suggestion with context and explanation."""
    file_path: str
    title: str = ""  # Brief title like "Add interface implementation"
    explanation: str = ""  # Why this change is needed
    
    # For new files
    is_new_file: bool = False
    new_file_content: str = ""
    
    # For modifications to existing files
    changes: List[CodeChange] = Field(default_factory=list)
    
    # Context for humans
    context_before: str = ""  # Lines before the change area
    context_after: str = ""   # Lines after the change area
    
    # Validation status
    confidence: float = 1.0  # 0-1 confidence this will work
    validation_notes: List[str] = Field(default_factory=list)
    
    # Operator guidance
    manual_steps: List[str] = Field(default_factory=list)  # Step-by-step instructions
    copy_paste_ready: str = ""  # Code ready to copy-paste
    testing_notes: str = ""  # How to verify this change works
    rollback_instructions: str = ""  # How to undo if needed
    
    # Failure prevention
    common_mistakes: List[str] = Field(default_factory=list)  # Pitfalls to avoid
    prerequisites: List[str] = Field(default_factory=list)  # What must be done first
    
    @field_validator("changes", mode="before")
    @classmethod
    def convert_change_dicts(cls, v):
        if isinstance(v, list):
            return [CodeChange(**c) if isinstance(c, dict) else c for c in v]
        return v


class SuggestionValidation(BaseModel):
    """Light validation results for suggestion mode."""
    is_valid: bool = True
    cycle_logically_broken: bool = False
    types_exist: bool = True  # All referenced types exist in source or suggestions
    no_hallucinations_detected: bool = True
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class CycleContext(BaseModel):
    """Context about the cycle being addressed - helps operators understand the problem."""
    nodes: List[str] = Field(default_factory=list)  # Files/modules in the cycle
    edges: List[Dict[str, str]] = Field(default_factory=list)  # Dependency edges
    cycle_type: str = ""  # "bidirectional", "transitive", "multi-node"
    hotspot_file: str = ""  # The file that appears in most dependencies
    breaking_edge: str = ""  # The specific edge to break (if identified)
    dependency_description: str = ""  # Human-readable description of the cycle


class FailurePattern(BaseModel):
    """Detected failure pattern with remediation guidance."""
    pattern_type: str = ""  # "search_mismatch", "hallucination", "syntax_error", etc.
    description: str = ""
    affected_files: List[str] = Field(default_factory=list)
    likely_cause: str = ""
    remediation: str = ""  # What to do about it
    evidence: List[str] = Field(default_factory=list)  # Specific evidence


class OperatorGuidance(BaseModel):
    """Actionable guidance for the human operator."""
    executive_summary: str = ""  # TL;DR for busy operators
    difficulty_rating: str = "easy"  # "easy", "moderate", "complex", "expert"
    estimated_time: str = ""  # "5 minutes", "30 minutes", etc.
    prerequisites: List[str] = Field(default_factory=list)  # What to do before starting
    step_by_step: List[str] = Field(default_factory=list)  # Ordered instructions
    verification_steps: List[str] = Field(default_factory=list)  # How to verify success
    common_pitfalls: List[str] = Field(default_factory=list)  # Things that often go wrong
    when_to_escalate: str = ""  # When to ask for help
    alternative_approaches: List[str] = Field(default_factory=list)  # Other ways to solve


class DiagnosticInfo(BaseModel):
    """Diagnostic information for troubleshooting."""
    llm_model: str = ""
    llm_provider: str = ""
    prompt_tokens: int = 0
    response_tokens: int = 0
    generation_time_ms: int = 0
    pipeline_version: str = ""
    timestamp: str = ""
    config_snapshot: Dict[str, Any] = Field(default_factory=dict)  # Relevant config


class SuggestionReport(BaseModel):
    """Complete suggestion report - the output of suggestion mode.
    
    This is designed to be a ONE-STOP-SHOP for operators, containing:
    - What the cycle is and why it's problematic
    - What the LLM suggests doing about it
    - Step-by-step instructions for manual application
    - Failure patterns and how to avoid them
    - Verification and rollback guidance
    
    Note: This differs from RefactorRoadmap which is a POST-MORTEM of an 
    actual refactoring attempt (what succeeded, what failed). SuggestionReport
    is BEFORE any changes are applied - it's the plan, not the execution report.
    """
    cycle_id: str = ""
    strategy: str = ""
    strategy_rationale: str = ""
    
    # Overall assessment
    confidence: float = 0.0
    cycle_will_be_broken: bool = False
    
    # The suggestions themselves
    suggestions: List[RefactorSuggestion] = Field(default_factory=list)
    
    # Light validation
    validation: Optional[SuggestionValidation] = None
    
    # Execution order recommendation
    suggested_order: List[str] = Field(default_factory=list)  # File paths in order
    
    # Human notes
    notes: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # Raw LLM output for debugging
    llm_response: str = ""
    
    # === NEW: Operator-friendly additions ===
    
    # Cycle context - understand the problem
    cycle_context: Optional[CycleContext] = None
    
    # Failure patterns detected (from validation or LLM analysis)
    failure_patterns: List[FailurePattern] = Field(default_factory=list)
    
    # Actionable operator guidance
    operator_guidance: Optional[OperatorGuidance] = None
    
    # Diagnostic info for troubleshooting
    diagnostics: Optional[DiagnosticInfo] = None
    
    # Quick-reference sections
    files_to_create: List[str] = Field(default_factory=list)  # New files
    files_to_modify: List[str] = Field(default_factory=list)  # Existing files
    imports_to_add: List[Dict[str, str]] = Field(default_factory=list)  # {file, import}
    imports_to_remove: List[Dict[str, str]] = Field(default_factory=list)  # {file, import}
    
    @field_validator("suggestions", mode="before")
    @classmethod
    def convert_suggestion_dicts(cls, v):
        if isinstance(v, list):
            return [RefactorSuggestion(**s) if isinstance(s, dict) else s for s in v]
        return v
    
    @field_validator("failure_patterns", mode="before")
    @classmethod
    def convert_failure_pattern_dicts(cls, v):
        if isinstance(v, list):
            return [FailurePattern(**p) if isinstance(p, dict) else p for p in v]
        return v
    
    def to_markdown(self) -> str:
        """Generate a comprehensive markdown report for human operators.
        
        This is designed to be a ONE-STOP-SHOP containing everything an
        operator needs to understand and implement the refactoring.
        """
        lines = []
        
        # =================================================================
        # HEADER & EXECUTIVE SUMMARY
        # =================================================================
        lines.extend([
            f"# 🔄 Refactoring Suggestions: {self.cycle_id}",
            "",
        ])
        
        # Executive summary if available
        if self.operator_guidance and self.operator_guidance.executive_summary:
            lines.extend([
                "## 📋 Executive Summary",
                "",
                self.operator_guidance.executive_summary,
                "",
                f"**Difficulty:** {self.operator_guidance.difficulty_rating.upper()}",
                f"**Estimated Time:** {self.operator_guidance.estimated_time or 'Not estimated'}",
                "",
            ])
        
        # Quick stats
        lines.extend([
            "## 📊 Overview",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Strategy | {self.strategy or 'Not specified'} |",
            f"| Confidence | {self.confidence:.0%} |",
            f"| Cycle Broken | {'✅ Yes' if self.cycle_will_be_broken else '❓ Needs verification'} |",
            f"| Files to Create | {len(self.files_to_create)} |",
            f"| Files to Modify | {len(self.files_to_modify)} |",
            f"| Total Suggestions | {len(self.suggestions)} |",
            "",
        ])
        
        if self.strategy_rationale:
            lines.extend([
                f"**Rationale:** {self.strategy_rationale}",
                "",
            ])
        
        # =================================================================
        # CYCLE CONTEXT - Understanding the Problem
        # =================================================================
        if self.cycle_context:
            ctx = self.cycle_context
            lines.extend([
                "---",
                "## 🔍 Understanding the Cycle",
                "",
            ])
            
            if ctx.dependency_description:
                lines.extend([ctx.dependency_description, ""])
            
            if ctx.nodes:
                lines.append("**Files Involved:**")
                for node in ctx.nodes:
                    lines.append(f"- `{node}`")
                lines.append("")
            
            if ctx.cycle_type:
                lines.append(f"**Cycle Type:** {ctx.cycle_type}")
            if ctx.hotspot_file:
                lines.append(f"**Hotspot File:** `{ctx.hotspot_file}` (appears in most dependencies)")
            if ctx.breaking_edge:
                lines.append(f"**Edge to Break:** {ctx.breaking_edge}")
            lines.append("")
            
            if ctx.edges:
                lines.extend([
                    "**Dependency Graph:**",
                    "```",
                ])
                for edge in ctx.edges:
                    source = edge.get("source", edge.get("from", "?"))
                    target = edge.get("target", edge.get("to", "?"))
                    lines.append(f"  {source} → {target}")
                lines.extend(["```", ""])
        
        # =================================================================
        # QUICK REFERENCE - What Changes at a Glance
        # =================================================================
        lines.extend([
            "---",
            "## ⚡ Quick Reference",
            "",
        ])
        
        if self.files_to_create:
            lines.append("**New Files to Create:**")
            for f in self.files_to_create:
                lines.append(f"- 📄 `{f}`")
            lines.append("")
        
        if self.files_to_modify:
            lines.append("**Files to Modify:**")
            for f in self.files_to_modify:
                lines.append(f"- ✏️ `{f}`")
            lines.append("")
        
        if self.imports_to_add:
            lines.append("**Imports to Add:**")
            for imp in self.imports_to_add:
                lines.append(f"- In `{imp.get('file', '?')}`: `{imp.get('import', '?')}`")
            lines.append("")
        
        if self.imports_to_remove:
            lines.append("**Imports to Remove:**")
            for imp in self.imports_to_remove:
                lines.append(f"- From `{imp.get('file', '?')}`: ~~`{imp.get('import', '?')}`~~")
            lines.append("")
        
        # =================================================================
        # RECOMMENDED ORDER
        # =================================================================
        if self.suggested_order:
            lines.extend([
                "---",
                "## 📝 Recommended Order of Changes",
                "",
                "Apply changes in this order to minimize issues:",
                "",
            ])
            for i, path in enumerate(self.suggested_order, 1):
                lines.append(f"{i}. `{path}`")
            lines.append("")
        
        # =================================================================
        # OPERATOR GUIDANCE - Step by Step
        # =================================================================
        if self.operator_guidance:
            og = self.operator_guidance
            
            if og.prerequisites:
                lines.extend([
                    "---",
                    "## ⚠️ Before You Start",
                    "",
                ])
                for prereq in og.prerequisites:
                    lines.append(f"- [ ] {prereq}")
                lines.append("")
            
            if og.step_by_step:
                lines.extend([
                    "---",
                    "## 📋 Step-by-Step Instructions",
                    "",
                ])
                for i, step in enumerate(og.step_by_step, 1):
                    lines.append(f"{i}. {step}")
                lines.append("")
            
            if og.common_pitfalls:
                lines.extend([
                    "### ⚠️ Common Pitfalls to Avoid",
                    "",
                ])
                for pitfall in og.common_pitfalls:
                    lines.append(f"- ❌ {pitfall}")
                lines.append("")
        
        # =================================================================
        # VALIDATION STATUS
        # =================================================================
        if self.validation:
            lines.extend([
                "---",
                "## ✅ Validation Status",
                "",
            ])
            if self.validation.cycle_logically_broken:
                lines.append("- ✅ Cycle logically broken by these changes")
            if self.validation.types_exist:
                lines.append("- ✅ All referenced types exist")
            if self.validation.no_hallucinations_detected:
                lines.append("- ✅ No hallucinations detected")
            for warning in self.validation.warnings:
                lines.append(f"- ⚠️ {warning}")
            for error in self.validation.errors:
                lines.append(f"- ❌ {error}")
            lines.append("")
        
        # =================================================================
        # FAILURE PATTERNS (if any detected)
        # =================================================================
        if self.failure_patterns:
            lines.extend([
                "---",
                "## 🚨 Detected Issues & Remediation",
                "",
            ])
            for pattern in self.failure_patterns:
                lines.extend([
                    f"### {pattern.pattern_type.replace('_', ' ').title()}",
                    "",
                    f"**Description:** {pattern.description}",
                    "",
                ])
                if pattern.affected_files:
                    lines.append("**Affected Files:**")
                    for f in pattern.affected_files[:5]:
                        lines.append(f"- `{f}`")
                    lines.append("")
                if pattern.likely_cause:
                    lines.append(f"**Likely Cause:** {pattern.likely_cause}")
                    lines.append("")
                if pattern.remediation:
                    lines.append(f"**How to Fix:** {pattern.remediation}")
                    lines.append("")
                if pattern.evidence:
                    lines.append("**Evidence:**")
                    for ev in pattern.evidence[:3]:
                        lines.append(f"- {ev}")
                    lines.append("")
        
        # =================================================================
        # DETAILED SUGGESTIONS
        # =================================================================
        lines.extend([
            "---",
            "## 📝 Detailed Suggestions",
            "",
        ])
        
        for i, suggestion in enumerate(self.suggestions, 1):
            lines.extend([
                "---",
                f"### Suggestion {i}: {suggestion.title}",
                "",
                f"**File:** `{suggestion.file_path}`",
                f"**Confidence:** {suggestion.confidence:.0%}",
                "",
            ])
            
            if suggestion.explanation:
                lines.extend([f"**Explanation:** {suggestion.explanation}", ""])
            
            # Prerequisites for this suggestion
            if suggestion.prerequisites:
                lines.append("**Prerequisites:**")
                for prereq in suggestion.prerequisites:
                    lines.append(f"- {prereq}")
                lines.append("")
            
            # The actual changes
            if suggestion.is_new_file:
                lines.extend([
                    "#### New File Content",
                    "",
                    "Create this file with the following content:",
                    "",
                    "```",
                    suggestion.new_file_content,
                    "```",
                    "",
                ])
                
                # Copy-paste ready
                if suggestion.copy_paste_ready:
                    lines.extend([
                        "**Copy-Paste Ready:**",
                        "```",
                        suggestion.copy_paste_ready,
                        "```",
                        "",
                    ])
            else:
                for j, change in enumerate(suggestion.changes, 1):
                    if len(suggestion.changes) > 1:
                        lines.append(f"#### Change {j}")
                    
                    if change.line_start > 0:
                        lines.append(f"**Location:** Lines {change.line_start}-{change.line_end}")
                    
                    if change.why_needed:
                        lines.append(f"**Why:** {change.why_needed}")
                    
                    lines.extend([
                        "",
                        "**Find this code:**",
                        "```",
                        change.original_code,
                        "```",
                        "",
                        "**Replace with:**",
                        "```",
                        change.suggested_code,
                        "```",
                        "",
                    ])
                    
                    if change.potential_issues:
                        lines.append("**⚠️ Watch out for:**")
                        for issue in change.potential_issues:
                            lines.append(f"- {issue}")
                        lines.append("")
                    
                    if change.dependencies:
                        lines.append("**Dependencies:**")
                        for dep in change.dependencies:
                            lines.append(f"- {dep}")
                        lines.append("")
            
            # Manual steps
            if suggestion.manual_steps:
                lines.append("**Step-by-Step:**")
                for step in suggestion.manual_steps:
                    lines.append(f"1. {step}")
                lines.append("")
            
            # Testing notes
            if suggestion.testing_notes:
                lines.extend([
                    "**How to Verify:**",
                    suggestion.testing_notes,
                    "",
                ])
            
            # Common mistakes for this suggestion
            if suggestion.common_mistakes:
                lines.append("**Common Mistakes:**")
                for mistake in suggestion.common_mistakes:
                    lines.append(f"- ❌ {mistake}")
                lines.append("")
            
            # Validation notes
            if suggestion.validation_notes:
                lines.append("**Validation Notes:**")
                for note in suggestion.validation_notes:
                    lines.append(f"- {note}")
                lines.append("")
            
            # Rollback
            if suggestion.rollback_instructions:
                lines.extend([
                    "**Rollback Instructions:**",
                    suggestion.rollback_instructions,
                    "",
                ])
        
        # =================================================================
        # VERIFICATION STEPS
        # =================================================================
        if self.operator_guidance and self.operator_guidance.verification_steps:
            lines.extend([
                "---",
                "## ✅ Verification Steps",
                "",
                "After applying all changes, verify success:",
                "",
            ])
            for i, step in enumerate(self.operator_guidance.verification_steps, 1):
                lines.append(f"{i}. [ ] {step}")
            lines.append("")
        
        # =================================================================
        # WARNINGS AND NOTES
        # =================================================================
        if self.warnings:
            lines.extend([
                "---",
                "## ⚠️ Warnings",
                "",
            ])
            for warning in self.warnings:
                lines.append(f"- ⚠️ {warning}")
            lines.append("")
        
        if self.notes:
            lines.extend([
                "## 📌 Additional Notes",
                "",
            ])
            for note in self.notes:
                lines.append(f"- {note}")
            lines.append("")
        
        # =================================================================
        # ALTERNATIVE APPROACHES
        # =================================================================
        if self.operator_guidance and self.operator_guidance.alternative_approaches:
            lines.extend([
                "---",
                "## 🔀 Alternative Approaches",
                "",
                "If the suggested approach doesn't work, consider:",
                "",
            ])
            for alt in self.operator_guidance.alternative_approaches:
                lines.append(f"- {alt}")
            lines.append("")
        
        # When to escalate
        if self.operator_guidance and self.operator_guidance.when_to_escalate:
            lines.extend([
                "## 🆘 When to Escalate",
                "",
                self.operator_guidance.when_to_escalate,
                "",
            ])
        
        # =================================================================
        # DIAGNOSTICS (collapsible for troubleshooting)
        # =================================================================
        if self.diagnostics:
            diag = self.diagnostics
            lines.extend([
                "---",
                "<details>",
                "<summary>🔧 Diagnostics (click to expand)</summary>",
                "",
                "| Property | Value |",
                "|----------|-------|",
                f"| LLM Model | {diag.llm_model or 'N/A'} |",
                f"| LLM Provider | {diag.llm_provider or 'N/A'} |",
                f"| Prompt Tokens | {diag.prompt_tokens} |",
                f"| Response Tokens | {diag.response_tokens} |",
                f"| Generation Time | {diag.generation_time_ms}ms |",
                f"| Pipeline Version | {diag.pipeline_version or 'N/A'} |",
                f"| Timestamp | {diag.timestamp or 'N/A'} |",
                "",
            ])
            if diag.config_snapshot:
                lines.append("**Config Snapshot:**")
                lines.append("```json")
                import json
                lines.append(json.dumps(diag.config_snapshot, indent=2))
                lines.append("```")
            lines.extend([
                "",
                "</details>",
                "",
            ])
        
        # =================================================================
        # RAW LLM RESPONSE (collapsible for debugging)
        # =================================================================
        if self.llm_response:
            lines.extend([
                "<details>",
                "<summary>🤖 Raw LLM Response (click to expand)</summary>",
                "",
                "```",
                self.llm_response[:5000],
            ])
            if len(self.llm_response) > 5000:
                lines.append(f"... ({len(self.llm_response) - 5000} more characters)")
            lines.extend([
                "```",
                "",
                "</details>",
            ])
        
        return "\n".join(lines)

from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from .agent_base import Agent, AgentResult
from .llm_utils import call_llm
from utils.snippet_selector import select_relevant_snippet, build_file_snippets_with_priority
from utils.prompt_loader import load_template, safe_format
from utils.logging import get_logger
from utils.rag_query_builder import RAGQueryBuilder, QueryIntent
from utils.context_budget import (
    BudgetCategory, TokenEstimator,
    prioritize_cycle_files, get_file_budget, create_budget_for_agent,
    truncate_to_token_budget
)
from utils.syntax_checker import (
    validate_code_block, check_truncation, check_introduced_issues,
    get_common_indent, normalize_line_endings, SyntaxIssue
)
from utils.syntax_repair import auto_repair_syntax, should_attempt_repair, RepairResult
from utils.compile_checker import CompileChecker, CompileResult
from utils.patch_parser import (
    parse_json_patches, parse_marker_patches, parse_search_replace_json,
    infer_patches, clean_code_content, extract_patches_from_data
)
from utils.patch_applier import (
    SearchReplaceResult, apply_search_replace_atomic, apply_search_replace_list_atomic,
    apply_with_partial_rollback, try_find_search_text, extract_line_hint
)
from utils.diff_utils import (
    make_unified_diff, looks_truncated, validate_patched_content,
    check_for_truncation, get_common_indent as diff_get_common_indent
)
from utils.llm_logger import (
    log_llm_call, log_llm_response, log_llm_summary, LLMCallTracker
)
# New mode imports
from utils.scaffolding import (
    run_scaffolding_phase, extract_scaffold_from_plan, ScaffoldResult
)
from utils.minimal_diff import (
    run_minimal_diff_mode, format_minimal_diff_summary, MinimalDiffResult
)
from utils.roadmap_builder import (
    RoadmapBuilder, classify_failure, build_roadmap_from_results
)
from utils.suggestion_builder import (
    build_suggestion_report, enrich_with_context
)
from utils.light_validator import (
    validate_suggestions, add_validation_to_report
)
from utils.line_patch import (
    add_line_numbers, build_numbered_file_snippets, 
    parse_line_patches, apply_line_patches
)
from utils.simple_format import (
    build_simple_format_prompt, parse_simple_format,
    convert_simple_to_line_patches, should_use_simple_format
)
from models.schemas import (
    CycleSpec, CycleDescription, RefactorProposal, Patch, ValidationReport, 
    RevertedFile, RefactorRoadmap, ScaffoldFile, PartialAttempt, SuggestionReport
)
from config import RefactorConfig
import json
import difflib
import re

logger = get_logger("refactor")


@dataclass
class CompileCheckInfo:
    """Information about compile/lint check results."""
    checked: bool = False
    success: bool = True
    tool_used: str = ""
    error_count: int = 0
    warning_count: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class AutoRepairInfo:
    """Information about auto-repair attempts."""
    attempted: bool = False
    was_repaired: bool = False
    repairs_made: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class PatchProcessingResult:
    """Result of processing a single file patch."""
    path: str
    original: str
    patched: str
    diff: str
    status: str  # "unchanged", "applied", "partial", "failed", "reverted"
    warnings: List[str]
    confidence: float
    applied_blocks: int
    total_blocks: int
    revert_reason: str
    pre_validated: bool
    validation_issues: List[str]
    has_critical_error: bool  # True if critical syntax/validation error found
    original_patched: Optional[str]  # Content before revert (if reverted)
    compile_info: Optional[CompileCheckInfo] = None  # Compile/lint check results
    repair_info: Optional[AutoRepairInfo] = None  # Auto-repair info


class RefactorAgent(Agent):
    """Proposes code patches to break cyclic dependencies.
    
    This agent:
    1. Extracts strategy hints from the Describer's output
    2. Queries RAG for implementation guidance (using conceptual terms)
    3. Includes mini-examples of refactoring patterns in prompts
    4. Uses chain-of-thought reasoning for complex cycles
    """
    
    name = "refactor"
    version = "0.7"

    # Mini-examples of refactoring patterns (language-agnostic concepts)
    PATTERN_EXAMPLES = {
        "interface_extraction": """
**Interface Extraction Pattern**
Before:
  ClassA imports ClassB (uses methods directly)
  ClassB imports ClassA (uses methods directly)
  
After:
  Create IClassB interface with methods ClassA needs
  ClassB implements IClassB
  ClassA depends on IClassB (not ClassB)
  Cycle broken: ClassB no longer needs to import ClassA
""",
        "dependency_inversion": """
**Dependency Inversion Pattern**
Before:
  HighLevelModule imports LowLevelModule directly
  
After:
  Create IService interface in HighLevelModule's layer
  LowLevelModule implements IService
  HighLevelModule depends on IService abstraction
  Dependency direction inverted
""",
        "shared_module": """
**Shared Module Extraction Pattern**
Before:
  ModuleA imports ModuleB (for shared utility)
  ModuleB imports ModuleA (for shared utility)
  
After:
  Create SharedModule with common functionality
  ModuleA imports SharedModule
  ModuleB imports SharedModule
  ModuleA and ModuleB no longer import each other
""",
        "mediator": """
**Mediator Pattern**
Before:
  ComponentA imports ComponentB (direct communication)
  ComponentB imports ComponentA (direct communication)
  
After:
  Create Mediator class
  ComponentA and ComponentB both depend on Mediator
  Mediator coordinates communication between them
  No direct dependencies between A and B
""",
    }

    def __init__(
        self,
        llm=None,
        prompt_template: str = None,
        prompt_template_compact: str = None,
        prompt_template_plan: str = None,
        prompt_template_plan_compact: str = None,
        prompt_template_file: str = None,
        prompt_template_file_compact: str = None,
        max_file_chars: int = 4000,
        rag_service=None,
        context_window: int = 4096,
        refactor_config: Optional[RefactorConfig] = None,
    ):
        self.llm = llm
        self.prompt_template = prompt_template
        self.prompt_template_compact = prompt_template_compact
        self.prompt_template_plan = prompt_template_plan
        self.prompt_template_plan_compact = prompt_template_plan_compact
        self.prompt_template_file = prompt_template_file
        self.prompt_template_file_compact = prompt_template_file_compact
        self.max_file_chars = max_file_chars
        self.rag_service = rag_service
        self.query_builder = RAGQueryBuilder()
        self.context_window = context_window
        self.refactor_config = refactor_config or RefactorConfig()
        
        # Initialize compile checker based on config
        self.compile_checker = CompileChecker(
            enabled=self.refactor_config.compile_check,
            timeout=self.refactor_config.compile_check_timeout,
        )
        
        # Determine if compact prompts should be used
        self.use_compact_prompts = self._should_use_compact_prompts()
        if self.use_compact_prompts:
            logger.info(f"Using compact prompts (context_window={context_window}, compact_enabled={self.refactor_config.compact_prompts})")
    
    def _should_use_compact_prompts(self) -> bool:
        """Determine whether to use compact prompts based on config and context window."""
        # Explicit config takes precedence
        if self.refactor_config.compact_prompts:
            return True
        
        # Auto-detect based on context window size
        threshold = getattr(self.refactor_config, 'auto_compact_threshold', 8192)
        if self.context_window < threshold:
            logger.debug(f"Auto-enabling compact prompts: context_window={self.context_window} < threshold={threshold}")
            return True
        
        return False
    
    def _get_prompt_template(self) -> Optional[str]:
        """Get the appropriate prompt template based on compact mode."""
        if self.use_compact_prompts and self.prompt_template_compact:
            logger.debug("Using compact prompt template")
            return self.prompt_template_compact
        return self.prompt_template

    def _should_use_sequential_mode(self, num_files: int) -> bool:
        """Determine whether to use sequential file mode."""
        # Explicit config takes precedence
        if self.refactor_config.sequential_file_mode:
            return True
        
        # Auto-detect based on number of files
        threshold = getattr(self.refactor_config, 'auto_sequential_threshold', 3)
        if num_files >= threshold:
            logger.debug(f"Auto-enabling sequential mode: {num_files} files >= threshold={threshold}")
            return True
        
        return False

    # =========================================================================
    # New Refactoring Modes
    # =========================================================================
    
    def _run_minimal_diff_mode(
        self,
        cycle_spec: CycleSpec,
        description: CycleDescription,
    ) -> AgentResult:
        """Run minimal diff mode - focuses on single smallest change.
        
        This mode:
        1. Identifies the weakest edge in the cycle
        2. Generates only the minimal change to break that edge
        3. Prioritizes simple changes over complex refactoring
        
        Returns:
            AgentResult with RefactorProposal (minimal)
        """
        logger.info("Running in MINIMAL DIFF MODE")
        
        # Build file snippets with full-file priority
        files_dict = [{"path": f.path, "content": f.content} for f in cycle_spec.files]
        cycle_dict = cycle_spec.model_dump()
        
        # Calculate budget for file content
        file_budget = int(self.context_window * 0.4 * 4)  # 40% of context, ~4 chars/token
        
        if self.refactor_config.prioritize_full_files:
            file_snippets, file_status = build_file_snippets_with_priority(
                files_dict,
                cycle_dict,
                file_budget,
                prioritize_full=True,
                max_chars_per_file=self.refactor_config.full_file_max_chars,
                full_file_budget_pct=self.refactor_config.full_file_budget_pct,
            )
            full_count = sum(1 for s in file_status.values() if s == 'full')
            logger.info(f"File priority: {full_count}/{len(file_status)} files included in full")
        else:
            snippets = []
            for f in cycle_spec.files:
                snippet = select_relevant_snippet(f.content or "", f.path, cycle_dict, self.max_file_chars)
                snippets.append(f"--- FILE: {f.path} ---\n{snippet}")
            file_snippets = "\n\n".join(snippets)
        
        # Run minimal diff
        def llm_call_fn(prompt):
            call_id = log_llm_call("refactor", "minimal_diff", prompt, {"cycle_id": cycle_spec.id})
            response = call_llm(self.llm, prompt)
            log_llm_response("refactor", "minimal_diff", response, call_id=call_id)
            return response
        
        result = run_minimal_diff_mode(
            cycle_dict,
            files_dict,
            file_snippets,
            llm_call_fn,
        )
        
        if not result.success:
            logger.warning("Minimal diff mode failed to generate patch")
            return AgentResult(
                status="error",
                output=None,
                logs=f"Minimal diff failed: {result.raw_llm_response[:500]}"
            )
        
        # Convert result to RefactorProposal
        patches = []
        if result.patch:
            patch_data = result.patch
            file_path = patch_data.get("path", "")
            
            # Find original content
            original = ""
            for f in cycle_spec.files:
                if f.path == file_path or f.path.endswith(file_path.split('/')[-1].split('\\')[-1]):
                    original = f.content or ""
                    break
            
            # Apply the patch
            patched = original
            sr_list = patch_data.get("search_replace", [])
            
            for sr in sr_list:
                search = sr.get("search", "")
                replace = sr.get("replace", "")
                if search and search in patched:
                    patched = patched.replace(search, replace, 1)
            
            diff = make_unified_diff(original, patched, file_path)
            
            patches.append(Patch(
                path=file_path,
                original=original,
                patched=patched,
                diff=diff,
                status="applied" if patched != original else "unchanged",
                confidence=result.confidence,
            ))
        
        # Handle new file if needed
        if result.new_file and result.new_file.get("needed"):
            new_path = result.new_file.get("path", "")
            new_content = result.new_file.get("content", "")
            patches.append(Patch(
                path=new_path,
                original="",
                patched=new_content,
                diff=f"+++ {new_path}\n{new_content}",
                status="new_file",
            ))
        
        rationale = f"Minimal diff: {result.strategy}\nTarget: {result.target_edge}\n{result.rationale}"
        
        proposal = RefactorProposal(
            patches=patches,
            rationale=rationale,
            llm_response=result.raw_llm_response,
        )
        
        logger.info(f"Minimal diff complete: {len(patches)} patches, strategy={result.strategy}")
        return AgentResult(status="success", output=proposal.model_dump())

    def _run_suggestion_mode(
        self,
        cycle_spec: CycleSpec,
        description: CycleDescription,
    ) -> AgentResult:
        """Run suggestion mode - outputs human-reviewable suggestions without applying patches.
        
        This mode:
        1. Uses a special prompt optimized for suggestion output
        2. Builds suggestions with line numbers and context
        3. Performs light semantic validation (no strict syntax checks)
        4. Outputs a markdown-friendly report for review
        
        Returns:
            AgentResult with SuggestionReport as output
        """
        logger.info("Running in SUGGESTION MODE")
        
        # Build file snippets with full-file priority
        files_dict = [{"path": f.path, "content": f.content} for f in cycle_spec.files]
        cycle_dict = cycle_spec.model_dump()
        
        # Calculate budget for file content - give more budget since we're not applying patches
        file_budget = int(self.context_window * 0.5 * 4)  # 50% of context, ~4 chars/token
        
        if self.refactor_config.prioritize_full_files:
            file_snippets, file_status = build_file_snippets_with_priority(
                files_dict,
                cycle_dict,
                file_budget,
                prioritize_full=True,
                max_chars_per_file=self.refactor_config.full_file_max_chars,
                full_file_budget_pct=self.refactor_config.full_file_budget_pct,
            )
            full_count = sum(1 for s in file_status.values() if s == 'full')
            logger.info(f"File priority: {full_count}/{len(file_status)} files included in full")
        else:
            snippets = []
            for f in cycle_spec.files:
                snippet = select_relevant_snippet(f.content or "", f.path, cycle_dict, self.max_file_chars)
                snippets.append(f"--- FILE: {f.path} ---\n{snippet}")
            file_snippets = "\n\n".join(snippets)
        
        # Extract strategy hint from description
        strategy = self._extract_strategy_from_description(description)
        
        # Get RAG context
        rag_context = self._get_rag_context(cycle_spec, strategy, description)
        
        # Get pattern example
        pattern_example = self._get_pattern_example(strategy)
        
        # Load suggestion mode prompt
        suggestion_template = load_template("prompts/prompt_suggestion.txt")
        if suggestion_template:
            prompt_text = safe_format(
                suggestion_template,
                id=cycle_spec.id,
                graph=json.dumps(cycle_spec.graph.model_dump()),
                files=", ".join(cycle_spec.get_file_paths()),
                description=description.text,
                file_snippets=file_snippets,
                rag_context=rag_context or "(no reference materials available)",
                pattern_example=pattern_example,
                strategy=strategy or "not specified - choose the most appropriate",
            )
        else:
            # Fallback inline prompt
            prompt_text = self._build_suggestion_prompt_fallback(
                cycle_spec, description, file_snippets, strategy, pattern_example
            )
        
        # Enforce prompt budget
        prompt_text = self._enforce_prompt_budget(prompt_text, file_snippets, cycle_spec.get_file_paths())
        
        logger.debug(f"Built suggestion prompt with {len(prompt_text)} chars")
        
        try:
            # Log LLM input
            call_id = log_llm_call(
                "refactor", "suggestion_mode",
                prompt_text,
                context={"cycle_id": cycle_spec.id, "mode": "suggestion"}
            )
            
            llm_start = datetime.now()
            llm_response = call_llm(self.llm, prompt_text)
            llm_duration = (datetime.now() - llm_start).total_seconds() * 1000
            
            # Log LLM output
            response_str = str(llm_response) if isinstance(llm_response, str) else json.dumps(llm_response)
            log_llm_response(
                "refactor", "suggestion_mode",
                llm_response,
                call_id=call_id,
                duration_ms=llm_duration
            )
            
            logger.debug(f"LLM response received: {len(response_str)} chars")
            
            # Build suggestion report from LLM response
            context_lines = self.refactor_config.suggestion_context_lines
            
            # Get LLM model/provider info for diagnostics
            llm_model = getattr(self.llm, 'model', '') or getattr(self.llm, 'model_name', '') or ''
            llm_provider = getattr(self.llm, 'provider', '') or ''
            if hasattr(self.llm, '__class__'):
                llm_provider = llm_provider or self.llm.__class__.__name__
            
            report = build_suggestion_report(
                cycle_spec=cycle_spec,  # Pass the CycleSpec object, not dict
                llm_response=response_str,
                strategy=strategy or "",  # Ensure not None for Pydantic
                context_lines=context_lines,
                llm_model=llm_model,
                llm_provider=llm_provider,
                generation_time_ms=int(llm_duration),
            )
            
            # Add validation
            report = add_validation_to_report(report, cycle_spec)
            
            # Enrich with surrounding context from source files
            report.suggestions = enrich_with_context(report.suggestions, cycle_spec, context_lines)
            
            logger.info(f"Suggestion mode complete: {len(report.suggestions)} suggestions, "
                       f"cycle_will_be_broken={report.cycle_will_be_broken}, "
                       f"valid={report.validation.is_valid if report.validation else 'not validated'}")
            
            # Output as markdown or JSON based on config
            output_format = self.refactor_config.suggestion_output_format
            if output_format == "markdown":
                output = {
                    "mode": "suggestion",
                    "markdown": report.to_markdown(),
                    "report": report.model_dump(),
                }
            else:
                output = {
                    "mode": "suggestion",
                    "report": report.model_dump(),
                }
            
            return AgentResult(status="success", output=output)
            
        except Exception as e:
            logger.error(f"Suggestion mode failed: {e}", exc_info=True)
            return AgentResult(status="error", output=None, logs=f"Suggestion mode failed: {e}")
    
    def _build_suggestion_prompt_fallback(
        self,
        cycle_spec: CycleSpec,
        description: CycleDescription,
        file_snippets: str,
        strategy: Optional[str],
        pattern_example: str,
    ) -> str:
        """Build fallback suggestion prompt if template not found."""
        return f"""You are an expert software engineer. Propose refactoring changes for human review.

## Cycle Information
- ID: {cycle_spec.id}
- Graph: {json.dumps(cycle_spec.graph.model_dump())}
- Files: {', '.join(cycle_spec.get_file_paths())}

## Problem Description
{description.text}

## Recommended Strategy: {strategy or 'Choose the most appropriate'}

{pattern_example}

## Source Code
{file_snippets}

## Task
Output a JSON with:
- summary: overview of the refactoring approach
- strategy_used: the strategy you're using
- why_this_breaks_the_cycle: explanation
- new_files: array of new files to create (path, purpose, content)
- modifications: array of file changes (path, changes array with what/why/search/replace)
- suggested_order: implementation order
- risks_and_notes: things to verify
- confidence: high/medium/low

For `search` text, include 3+ lines of context. Be specific about WHAT and WHY for each change.
Output ONLY the JSON.
"""

    def _run_line_patch_mode(
        self,
        cycle_spec: CycleSpec,
        description: CycleDescription,
        validator_feedback: Optional[ValidationReport] = None,
    ) -> AgentResult:
        """Run line-based patching mode - uses line numbers instead of SEARCH/REPLACE.
        
        This mode:
        1. Includes line numbers in file snippets
        2. Asks LLM to output patches with line ranges
        3. Applies patches by line number (more reliable than text matching)
        
        Benefits:
        - No fuzzy text matching needed
        - Works even with whitespace differences
        - Better error messages with exact line numbers
        - Handles partial LLM output gracefully
        
        Args:
            cycle_spec: The cycle specification
            description: The cycle description from describer
            validator_feedback: Optional feedback from previous validation (for retries)
        
        Returns:
            AgentResult with RefactorProposal
        """
        is_retry = validator_feedback is not None
        logger.info(f"Running in LINE-BASED PATCH MODE (retry={is_retry})")
        
        # Build file dicts with content
        files_dict = [{"path": f.path, "content": f.content} for f in cycle_spec.files]
        cycle_dict = cycle_spec.model_dump()
        
        # Calculate budget for file content
        file_budget = int(self.context_window * 0.45 * 4)  # 45% of context, ~4 chars/token
        
        # Build numbered file snippets
        numbered_snippets, line_counts = build_numbered_file_snippets(
            files_dict,
            max_chars_per_file=file_budget // max(len(files_dict), 1),
        )
        
        logger.info(f"Built numbered snippets for {len(line_counts)} files")
        for path, count in line_counts.items():
            logger.debug(f"  {path}: {count} lines")
        
        # Extract strategy hint from description
        strategy = self._extract_strategy_from_description(description)
        
        # Get pattern example (brief for this mode)
        pattern_example = self._get_pattern_example(strategy)
        
        # Build feedback section if this is a retry
        feedback_section = ""
        if validator_feedback:
            feedback_issues = []
            for issue in validator_feedback.issues[:5]:  # Limit to top 5 issues
                if hasattr(issue, 'comment'):
                    feedback_issues.append(f"- {issue.comment}")
                elif isinstance(issue, dict):
                    feedback_issues.append(f"- {issue.get('comment', str(issue))}")
                else:
                    feedback_issues.append(f"- {str(issue)}")
            
            feedback_section = f"""
## IMPORTANT: Previous Attempt Failed

Your previous patch attempt had validation issues. Please address these:

{chr(10).join(feedback_issues)}

Ensure your line numbers are exact and match the numbered content below.
"""
            logger.info(f"Including {len(feedback_issues)} validation issues in retry prompt")
        
        # Load line-patch prompt template
        line_patch_template = load_template("prompts/prompt_line_patch.txt")
        if line_patch_template:
            prompt_text = safe_format(
                line_patch_template,
                cycle=json.dumps({
                    "id": cycle_spec.id,
                    "nodes": cycle_spec.graph.nodes,
                    "edges": cycle_spec.graph.edges,
                }),
                target_file=cycle_spec.files[0].path if cycle_spec.files else "unknown",
                patterns=pattern_example,
                file_content=numbered_snippets,
                feedback=feedback_section,  # Add feedback placeholder
            )
        else:
            # Fallback inline prompt
            prompt_text = self._build_line_patch_prompt_fallback(
                cycle_spec, description, numbered_snippets, strategy, pattern_example,
                feedback_section=feedback_section
            )
        
        # Enforce prompt budget
        prompt_text = self._enforce_prompt_budget(prompt_text, numbered_snippets, cycle_spec.get_file_paths())
        
        logger.debug(f"Built line-patch prompt with {len(prompt_text)} chars")
        
        try:
            # Log LLM input
            call_id = log_llm_call(
                "refactor", "line_patch_mode",
                prompt_text,
                context={"cycle_id": cycle_spec.id, "mode": "line_patch"}
            )
            
            llm_start = datetime.now()
            llm_response = call_llm(self.llm, prompt_text)
            llm_duration = (datetime.now() - llm_start).total_seconds() * 1000
            
            # Log LLM output
            response_str = str(llm_response) if isinstance(llm_response, str) else json.dumps(llm_response)
            log_llm_response(
                "refactor", "line_patch_mode",
                llm_response,
                call_id=call_id,
                duration_ms=llm_duration
            )
            
            logger.debug(f"LLM response received: {len(response_str)} chars")
            
            # Parse line-based patches from response
            raw_patches = parse_line_patches(response_str)
            logger.info(f"Parsed {len(raw_patches)} file patch(es) from LLM response")
            
            # Apply patches and build results
            patches = []
            for f in cycle_spec.files:
                # Find matching patch for this file
                file_patch = None
                for p in raw_patches:
                    patch_path = p.get("path", "")
                    if patch_path == f.path or f.path.endswith(patch_path.split('/')[-1]):
                        file_patch = p
                        break
                
                original = f.content or ""
                
                if file_patch:
                    # Apply line-based patches
                    result = apply_line_patches(original, [file_patch])
                    patched = result.content
                    
                    if result.success and result.applied_count > 0:
                        diff = self._make_unified_diff(original, patched, f.path)
                        patches.append(Patch(
                            path=f.path,
                            original=original,
                            patched=patched,
                            diff=diff,
                            status="applied",
                            confidence=1.0,
                            applied_blocks=result.applied_count,
                            total_blocks=result.total_count,
                            warnings=result.warnings,
                        ))
                        logger.info(f"Applied {result.applied_count} line patch(es) to {f.path}")
                    else:
                        # Failed to apply
                        patches.append(Patch(
                            path=f.path,
                            original=original,
                            patched=original,
                            diff="",
                            status="failed",
                            confidence=0.0,
                            warnings=result.errors + result.warnings,
                        ))
                        logger.warning(f"Failed to apply line patches to {f.path}: {result.errors}")
                else:
                    # No patch for this file
                    patches.append(Patch(
                        path=f.path,
                        original=original,
                        patched=original,
                        diff="",
                        status="unchanged",
                    ))
            
            # Extract reasoning from response
            reasoning = ""
            try:
                import json as json_mod
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_str)
                if json_match:
                    data = json_mod.loads(json_match.group(1))
                    reasoning = data.get("reasoning", "")
            except:
                pass
            
            proposal = RefactorProposal(
                patches=patches,
                rationale=f"Line-based patching: {reasoning or strategy or 'applied changes by line number'}",
                llm_response=response_str,
            )
            
            applied_count = sum(1 for p in patches if p.status == "applied")
            logger.info(f"Line-patch mode complete: {applied_count}/{len(patches)} files patched")
            
            return AgentResult(status="success", output=proposal.model_dump())
            
        except Exception as e:
            logger.error(f"Line-patch mode failed: {e}", exc_info=True)
            return AgentResult(status="error", output=None, logs=f"Line-patch mode failed: {e}")
    
    def _build_line_patch_prompt_fallback(
        self,
        cycle_spec: CycleSpec,
        description: CycleDescription,
        numbered_snippets: str,
        strategy: Optional[str],
        pattern_example: str,
        feedback_section: str = "",
    ) -> str:
        """Build fallback line-patch prompt if template not found."""
        return f"""You are refactoring code to break a cyclic dependency.

## Cycle Information
- ID: {cycle_spec.id}
- Nodes: {', '.join(cycle_spec.graph.nodes)}
- Edges: {json.dumps(cycle_spec.graph.edges)}

## Problem Description
{description.text}
{feedback_section}
## Recommended Strategy: {strategy or 'Choose the most appropriate'}

{pattern_example}

## Source Code (with line numbers)
{numbered_snippets}

## Output Format

Respond with JSON containing line-based patches:

```json
{{
  "patches": [
    {{
      "path": "path/to/file.ext",
      "changes": [
        {{
          "lines": [START_LINE, END_LINE],
          "new_content": "replacement code here",
          "description": "what this change does"
        }}
      ],
      "add_at_line": 1,
      "add_content": "import {{ X }} from './newLocation';"
    }}
  ],
  "reasoning": "Brief explanation"
}}
```

RULES:
1. Use EXACT line numbers from the snippets above
2. "lines": [START, END] replaces lines START through END (inclusive)
3. For insertions, use "add_at_line" and "add_content"
4. Include all replaced lines' content in "new_content"
5. Preserve indentation
"""

    def _run_simple_format_mode(
        self,
        cycle_spec: CycleSpec,
        description: CycleDescription,
    ) -> AgentResult:
        """Run simple format mode - text-based output for smaller LLMs (7B-14B).
        
        This mode uses a simpler text-based output format instead of JSON,
        which is easier for smaller models to generate correctly.
        
        Output format:
            STRATEGY: interface_extraction
            REASONING: Break cycle by extracting interface
            
            FILE: src/auth/AuthService.ts
            CHANGE: lines 1-2
            ---
            import { IUserProvider } from '../shared/IUserProvider';
            ---
            DESCRIPTION: Use interface import
        
        Returns:
            AgentResult with RefactorProposal
        """
        logger.info("Running in SIMPLE FORMAT MODE (for smaller LLMs)")
        
        # Build file dicts with content
        files_dict = [{"path": f.path, "content": f.content} for f in cycle_spec.files]
        cycle_dict = cycle_spec.model_dump()
        
        # Build numbered file snippets (reuse from line-patch mode)
        file_budget = int(self.context_window * 0.45 * 4)
        numbered_snippets, line_counts = build_numbered_file_snippets(
            files_dict,
            max_chars_per_file=file_budget // max(len(files_dict), 1),
        )
        
        logger.info(f"Built numbered snippets for {len(line_counts)} files")
        
        # Extract strategy hint from description
        strategy = self._extract_strategy_from_description(description)
        
        # Build simple format prompt
        prompt_text = build_simple_format_prompt(
            cycle_info=cycle_dict,
            file_snippets=numbered_snippets,
            strategy_hint=strategy,
        )
        
        # Enforce prompt budget
        prompt_text = self._enforce_prompt_budget(prompt_text, numbered_snippets, cycle_spec.get_file_paths())
        
        logger.debug(f"Built simple format prompt with {len(prompt_text)} chars")
        
        try:
            # Log LLM input
            call_id = log_llm_call(
                "refactor", "simple_format_mode",
                prompt_text,
                context={"cycle_id": cycle_spec.id, "mode": "simple_format"}
            )
            
            llm_start = datetime.now()
            llm_response = call_llm(self.llm, prompt_text)
            llm_duration = (datetime.now() - llm_start).total_seconds() * 1000
            
            # Log LLM output
            response_str = str(llm_response) if isinstance(llm_response, str) else json.dumps(llm_response)
            log_llm_response(
                "refactor", "simple_format_mode",
                llm_response,
                call_id=call_id,
                duration_ms=llm_duration
            )
            
            logger.debug(f"LLM response received: {len(response_str)} chars")
            
            # Parse simple format output
            simple_result = parse_simple_format(response_str)
            logger.info(f"Parsed {len(simple_result.changes)} changes, strategy: {simple_result.strategy}")
            
            # Log any parse warnings
            for warning in simple_result.parse_warnings:
                logger.warning(f"Parse warning: {warning}")
            
            # Convert to line patches and apply
            line_patches = convert_simple_to_line_patches(simple_result)
            
            # Apply patches and build results
            patches = []
            for f in cycle_spec.files:
                # Find matching patch for this file
                file_patch = None
                for p in line_patches:
                    patch_path = p.get("path", "")
                    if patch_path == f.path or f.path.endswith(patch_path.split('/')[-1]):
                        file_patch = p
                        break
                
                original = f.content or ""
                
                if file_patch:
                    if file_patch.get("is_new_file"):
                        # This shouldn't happen for existing files, skip
                        patches.append(Patch(
                            path=f.path,
                            original=original,
                            patched=original,
                            diff="",
                            status="unchanged",
                        ))
                    else:
                        # Apply line-based patches
                        result = apply_line_patches(original, [file_patch])
                        patched = result.content
                        
                        if result.success and result.applied_count > 0:
                            diff = self._make_unified_diff(original, patched, f.path)
                            patches.append(Patch(
                                path=f.path,
                                original=original,
                                patched=patched,
                                diff=diff,
                                status="applied",
                                confidence=1.0,
                                applied_blocks=result.applied_count,
                                total_blocks=result.total_count,
                                warnings=result.warnings,
                            ))
                            logger.info(f"Applied {result.applied_count} change(s) to {f.path}")
                        else:
                            patches.append(Patch(
                                path=f.path,
                                original=original,
                                patched=original,
                                diff="",
                                status="failed",
                                confidence=0.0,
                                warnings=result.errors + result.warnings,
                            ))
                            logger.warning(f"Failed to apply changes to {f.path}: {result.errors}")
                else:
                    patches.append(Patch(
                        path=f.path,
                        original=original,
                        patched=original,
                        diff="",
                        status="unchanged",
                    ))
            
            # Handle new files from simple format
            for p in line_patches:
                if p.get("is_new_file"):
                    new_path = p.get("path", "")
                    new_content = p.get("content", "")
                    patches.append(Patch(
                        path=new_path,
                        original="",
                        patched=new_content,
                        diff=f"+++ {new_path}\n{new_content[:500]}..." if len(new_content) > 500 else f"+++ {new_path}\n{new_content}",
                        status="new_file",
                        confidence=1.0,
                    ))
                    logger.info(f"Created new file: {new_path}")
            
            proposal = RefactorProposal(
                patches=patches,
                rationale=f"Simple format: {simple_result.strategy or 'auto'}. {simple_result.reasoning}",
                llm_response=response_str,
            )
            
            applied_count = sum(1 for p in patches if p.status in ("applied", "new_file"))
            logger.info(f"Simple format mode complete: {applied_count}/{len(patches)} files modified")
            
            return AgentResult(status="success", output=proposal.model_dump())
            
        except Exception as e:
            logger.error(f"Simple format mode failed: {e}", exc_info=True)
            return AgentResult(status="error", output=None, logs=f"Simple format mode failed: {e}")

    def _run_scaffolding_mode(
        self,
        cycle_spec: CycleSpec,
        description: CycleDescription,
        plan: Dict[str, Any],
    ) -> Tuple[ScaffoldResult, List[Dict[str, Any]]]:
        """Run scaffolding phase - creates interfaces/abstractions before modifying existing files.
        
        Args:
            cycle_spec: The cycle specification
            description: Cycle description from describer
            plan: The refactoring plan from planning phase
            
        Returns:
            Tuple of (ScaffoldResult, list of scaffold file info for inclusion in prompts)
        """
        logger.info("Running SCAFFOLDING PHASE")
        
        # Build file snippets for context
        files_dict = [{"path": f.path, "content": f.content} for f in cycle_spec.files]
        cycle_dict = cycle_spec.model_dump()
        
        snippets = []
        source_files = {}  # path -> content mapping for method extraction
        for f in cycle_spec.files:
            if f.content:
                source_files[f.path] = f.content
            snippet = select_relevant_snippet(f.content or "", f.path, cycle_dict, 2000)
            snippets.append(f"--- FILE: {f.path} ---\n{snippet}")
        file_snippets = "\n\n".join(snippets)
        
        # Run scaffolding with source files for enhanced method extraction
        def llm_call_fn(prompt):
            call_id = log_llm_call("refactor", "scaffolding", prompt, {"cycle_id": cycle_spec.id})
            response = call_llm(self.llm, prompt)
            log_llm_response("refactor", "scaffolding", response, call_id=call_id)
            return response
        
        scaffold_result = run_scaffolding_phase(
            plan,
            cycle_dict,
            file_snippets,
            llm_call_fn,
            validate=self.refactor_config.scaffolding_validate,
            source_files=source_files,  # Pass source files for method extraction
        )
        
        # Build scaffold info for subsequent prompts
        scaffold_context = []
        for sf in scaffold_result.files_created:
            scaffold_context.append({
                "path": sf.get("path", ""),
                "content": sf.get("content", ""),
                "purpose": sf.get("purpose", ""),
                "valid": sf.get("valid", False),
            })
        
        if scaffold_result.success:
            logger.info(f"Scaffolding complete: {len(scaffold_context)} files created")
        else:
            logger.warning(f"Scaffolding had issues: {scaffold_result.validation_errors}")
        
        return scaffold_result, scaffold_context

    def _build_file_snippets_with_priority(
        self,
        cycle_spec: CycleSpec,
        max_budget_chars: int,
    ) -> Tuple[str, Dict[str, str]]:
        """Build file snippets with priority for full file inclusion.
        
        When files are small enough, include them in full rather than snippets.
        This gives the LLM complete context and reduces hallucination.
        
        Args:
            cycle_spec: The cycle specification
            max_budget_chars: Maximum characters for all file content
            
        Returns:
            Tuple of (combined snippets string, dict of file path -> 'full'|'snippet')
        """
        if not self.refactor_config.prioritize_full_files:
            # Fall back to standard snippet selection
            snippets = []
            cycle_dict = cycle_spec.model_dump()
            for f in cycle_spec.files:
                snippet = select_relevant_snippet(f.content or "", f.path, cycle_dict, self.max_file_chars)
                snippets.append(f"--- FILE: {f.path} ---\n{snippet}")
            return "\n\n".join(snippets), {f.path: 'snippet' for f in cycle_spec.files}
        
        files_dict = [{"path": f.path, "content": f.content} for f in cycle_spec.files]
        cycle_dict = cycle_spec.model_dump()
        
        return build_file_snippets_with_priority(
            files_dict,
            cycle_dict,
            max_budget_chars,
            prioritize_full=True,
            max_chars_per_file=self.refactor_config.full_file_max_chars,
            full_file_budget_pct=self.refactor_config.full_file_budget_pct,
            snippet_max_chars=self.max_file_chars,
        )

    def _build_roadmap_output(
        self,
        cycle_spec: CycleSpec,
        strategy: str,
        patch_results: List[Dict[str, Any]],
        scaffold_result: Optional[ScaffoldResult] = None,
        minimal_diff_result: Optional[MinimalDiffResult] = None,
        llm_responses: List[str] = None,
    ) -> RefactorRoadmap:
        """Build a RefactorRoadmap for demo-friendly output.
        
        Args:
            cycle_spec: The cycle specification
            strategy: Strategy used
            patch_results: Results from patch processing
            scaffold_result: Results from scaffolding (if used)
            minimal_diff_result: Results from minimal diff (if used)
            llm_responses: Raw LLM responses
            
        Returns:
            RefactorRoadmap with full progress information
        """
        scaffold_data = None
        if scaffold_result:
            scaffold_data = {
                "files_created": scaffold_result.files_created,
                "validation_errors": scaffold_result.validation_errors,
            }
        
        minimal_diff_data = None
        if minimal_diff_result:
            minimal_diff_data = {
                "target_edge": minimal_diff_result.target_edge,
                "strategy": minimal_diff_result.strategy,
                "patch": minimal_diff_result.patch,
                "rationale": minimal_diff_result.rationale,
            }
        
        roadmap = build_roadmap_from_results(
            cycle_id=cycle_spec.id,
            strategy=strategy,
            patch_results=patch_results,
            scaffold_results=scaffold_data,
            minimal_diff_result=minimal_diff_data,
            llm_responses=llm_responses or [],
            cycle_spec=cycle_spec.model_dump(),
        )
        
        return roadmap

    
    def _get_plan_template(self) -> Optional[str]:
        """Get the planning prompt template."""
        if self.use_compact_prompts and self.prompt_template_plan_compact:
            return self.prompt_template_plan_compact
        return self.prompt_template_plan
    
    def _get_file_template(self) -> Optional[str]:
        """Get the per-file execution prompt template."""
        if self.use_compact_prompts and self.prompt_template_file_compact:
            return self.prompt_template_file_compact
        return self.prompt_template_file

    def _extract_strategy_from_description(self, description: CycleDescription) -> Optional[str]:
        """Extract the recommended strategy from the Describer's output."""
        text = description.text.lower()
        
        strategy_keywords = {
            "interface_extraction": ["interface extraction", "extract interface", "create interface", "iservice"],
            "dependency_inversion": ["dependency inversion", "invert dependency", "inversion of control"],
            "shared_module": ["shared module", "common module", "extract shared", "move to common"],
            "mediator": ["mediator", "coordinator", "event bus"],
        }
        
        for strategy, keywords in strategy_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    logger.info(f"Detected strategy hint from description: {strategy}")
                    return strategy
        
        # Default based on common patterns in text
        if "bidirectional" in text or "mutual" in text:
            return "interface_extraction"
        if "layer" in text or "violation" in text:
            return "dependency_inversion"
        
        return None
    
    def _get_rag_context(self, cycle: CycleSpec, strategy: Optional[str], description: CycleDescription) -> str:
        """Retrieve implementation-focused context from RAG."""
        if self.rag_service is None:
            logger.debug("RAG service not available, skipping context retrieval")
            return ""
        
        try:
            cycle_dict = cycle.model_dump()
            
            # Build implementation-focused queries
            queries = self.query_builder.build_queries_for_cycle(
                cycle_dict, 
                QueryIntent.IMPLEMENT,
                description.text  # Pass description as hints
            )
            
            # Add strategy-specific query if we know the strategy
            if strategy:
                strategy_queries = {
                    "interface_extraction": "how to extract interface break dependency example",
                    "dependency_inversion": "implementing dependency inversion principle example",
                    "shared_module": "extract common module reduce coupling example",
                    "mediator": "mediator pattern implementation example",
                }
                if strategy in strategy_queries:
                    queries.insert(0, strategy_queries[strategy])
            
            all_results = []
            seen_content = set()
            
            for query in queries[:3]:  # Limit queries
                logger.info(f"RAG Query: '{query}'")
                logger.info("Purpose: Find implementation guidance for refactoring")
                
                results = self.rag_service.query_with_scores(query, k=2)
                
                if results:
                    for doc, score in results:
                        content_hash = hash(doc.page_content[:200])
                        if content_hash not in seen_content:
                            seen_content.add(content_hash)
                            all_results.append((doc, score))
                            source = doc.metadata.get('source_file', 'unknown')
                            logger.info(f"  Retrieved: {source} (score: {score:.3f})")
            
            if all_results:
                logger.info(f"RAG total: {len(all_results)} unique document(s) for implementation")
                all_results.sort(key=lambda x: x[1])
                
                context_parts = []
                for doc, score in all_results[:3]:
                    source = doc.metadata.get('source_file', 'unknown')
                    context_parts.append(f"[{source}]\n{doc.page_content[:500]}")
                
                return "\n\n---\n\n".join(context_parts)
            else:
                logger.info("RAG: No implementation guidance found")
                
        except Exception as e:
            logger.warning(f"Failed to retrieve RAG context: {e}")
        
        return ""

    def _get_pattern_example(self, strategy: Optional[str]) -> str:
        """Get a mini-example for the recommended strategy."""
        if strategy and strategy in self.PATTERN_EXAMPLES:
            return self.PATTERN_EXAMPLES[strategy]
        
        # Return general examples if no specific strategy
        return """
**Common Refactoring Patterns:**

1. Interface Extraction: Create an interface that one module implements,
   and the other module depends on the interface instead of the concrete class.

2. Dependency Inversion: Ensure high-level modules don't depend on low-level modules.
   Both should depend on abstractions.

3. Extract Shared Module: Move common code to a new module that both can import,
   eliminating direct dependencies between the original modules.
"""

    # =========================================================================
    # Sequential File Mode - Two-phase approach
    # =========================================================================
    
    def _build_planning_prompt(
        self,
        cycle: CycleSpec,
        description: CycleDescription,
    ) -> str:
        """Build the planning prompt for Phase 1 of sequential mode."""
        file_paths = cycle.get_file_paths()
        cycle_dict = cycle.model_dump()
        
        # Build file snippets (abbreviated for planning)
        snippets = []
        for f in cycle.files:
            content = f.content or ""
            # For planning, include first 50 lines + last 20 lines to show structure
            lines = content.split('\n')
            if len(lines) > 80:
                snippet = '\n'.join(lines[:50]) + f"\n\n... ({len(lines) - 70} lines omitted) ...\n\n" + '\n'.join(lines[-20:])
            else:
                snippet = content[:self.max_file_chars]
            snippets.append(f"--- FILE: {f.path} ---\n{snippet}")
        
        file_snippets = "\n\n".join(snippets)
        
        # Load template
        template_path = self._get_plan_template()
        if template_path:
            tpl = load_template(template_path)
            prompt = safe_format(
                tpl,
                id=cycle.id,
                graph=json.dumps(cycle.graph.model_dump()),
                files=", ".join(file_paths),
                description=description.text,
                file_snippets=file_snippets,
            )
            return self._enforce_prompt_budget(prompt, file_snippets, file_paths)
        
        # Fallback inline template
        prompt = f"""Create a refactoring plan to break this cyclic dependency.

## Cycle: {cycle.id}
Graph: {json.dumps(cycle.graph.model_dump())}
Files: {', '.join(file_paths)}

## Description
{description.text}

## Source Code
{file_snippets}

Output a JSON plan with: strategy, summary, file_changes (per file), execution_order.
"""
        return self._enforce_prompt_budget(prompt, file_snippets, file_paths)

    def _build_per_file_prompt(
        self,
        file_path: str,
        file_content: str,
        plan: Dict[str, Any],
        previous_changes: List[Dict[str, Any]],
    ) -> str:
        """Build prompt for Phase 2 - patching a single file."""
        # Extract file-specific plan
        file_plan = "No specific plan for this file."
        for fc in plan.get("file_changes", []):
            if fc.get("path", "").endswith(file_path.split('/')[-1].split('\\')[-1]):
                file_plan = json.dumps(fc, indent=2)
                break
        
        # Format previous changes
        if previous_changes:
            prev_str = "\n".join([
                f"- {c.get('path', 'unknown')}: {', '.join(c.get('changes_made', ['changes applied']))}"
                for c in previous_changes
            ])
        else:
            prev_str = "No changes made yet (this is the first file)."
        
        # Load template
        template_path = self._get_file_template()
        if template_path:
            tpl = load_template(template_path)
            prompt = safe_format(
                tpl,
                plan=json.dumps(plan, indent=2),
                file_path=file_path,
                file_plan=file_plan,
                file_content=file_content,
                previous_changes=prev_str,
            )
            return self._enforce_prompt_budget(prompt, file_content, [file_path])
        
        # Fallback inline template
        prompt = f"""Implement the refactoring plan for this file.

## Plan
{json.dumps(plan, indent=2)}

## File: {file_path}
{file_plan}

## Content
```
{file_content}
```

## Previous Changes
{prev_str}

Output JSON with: path, search_replace (list of {{search, replace}}), changes_made.
Include 3+ lines of context in each SEARCH block.
"""
        return self._enforce_prompt_budget(prompt, file_content, [file_path])

    def _run_sequential_mode(
        self,
        cycle_spec: CycleSpec,
        description: CycleDescription,
        validator_feedback: Optional[ValidationReport] = None,
    ) -> AgentResult:
        """Run refactoring in sequential file mode (two-phase approach).
        
        Phase 1: Generate a plan describing all changes
        Phase 2: For each file, generate and apply patches
        
        Returns:
            AgentResult with RefactorProposal
        """
        logger.info("Running in SEQUENTIAL FILE MODE")
        
        # Track LLM call stats
        total_prompt_chars = 0
        total_response_chars = 0
        llm_call_count = 0
        start_time = datetime.now()
        
        # Phase 1: Planning
        logger.info("Phase 1: Generating refactoring plan...")
        plan_prompt = self._build_planning_prompt(cycle_spec, description)
        logger.debug(f"Planning prompt: {len(plan_prompt)} chars")
        
        try:
            # Log LLM input
            call_id = log_llm_call(
                "refactor", "planning",
                plan_prompt,
                context={"cycle_id": cycle_spec.id, "num_files": len(cycle_spec.files)}
            )
            total_prompt_chars += len(plan_prompt)
            llm_call_count += 1
            
            plan_start = datetime.now()
            plan_response = call_llm(self.llm, plan_prompt)
            plan_duration = (datetime.now() - plan_start).total_seconds() * 1000
            
            # Log LLM output
            response_str = str(plan_response)
            log_llm_response(
                "refactor", "planning",
                plan_response,
                call_id=call_id,
                duration_ms=plan_duration
            )
            total_response_chars += len(response_str)
            
            logger.debug(f"Plan response: {len(response_str)} chars")
            
            # Parse the plan
            plan = self._parse_plan_response(plan_response)
            if not plan:
                logger.error("Failed to parse planning response")
                return AgentResult(
                    status="error",
                    output=None,
                    logs="Failed to parse refactoring plan from LLM response"
                )
            
            strategy = plan.get("strategy", "unknown")
            logger.info(f"Plan generated: strategy={strategy}, files={len(plan.get('file_changes', []))}")
            
        except Exception as e:
            logger.error(f"Planning phase failed: {e}")
            return AgentResult(status="error", output=None, logs=f"Planning failed: {e}")
        
        # =====================================================================
        # Scaffolding Phase (if enabled and applicable)
        # =====================================================================
        scaffold_result = None
        scaffold_context = []
        
        if (self.refactor_config.scaffolding_mode and 
            strategy in ["interface_extraction", "dependency_inversion", "shared_module"] and
            plan.get("new_files")):
            
            logger.info("Running scaffolding phase before modifying existing files...")
            scaffold_result, scaffold_context = self._run_scaffolding_mode(
                cycle_spec, description, plan
            )
            
            if not scaffold_result.success and self.refactor_config.scaffolding_validate:
                logger.warning("Scaffolding validation failed, continuing with caution")
                # Don't block, but log the issues
                for error in scaffold_result.validation_errors:
                    logger.warning(f"Scaffold issue: {error}")
            
            # If scaffolding succeeded, use scaffold content instead of LLM generation
            if scaffold_result.success and scaffold_result.files_created:
                logger.info(f"Using {len(scaffold_result.files_created)} pre-validated scaffold files")
        
        # Determine execution order
        execution_order = plan.get("execution_order", [])
        if not execution_order:
            execution_order = [f.path for f in cycle_spec.files]
        
        # Phase 2: Per-file execution
        logger.info(f"Phase 2: Executing patches for {len(execution_order)} files...")
        
        all_inferred: List[Dict[str, Any]] = []
        previous_changes: List[Dict[str, Any]] = []
        llm_responses: List[str] = [str(plan_response)]
        
        # Handle new files - use scaffold files if available, otherwise generate
        for new_file in plan.get("new_files", []):
            new_path = new_file.get("path", "")
            if not new_path:
                continue
            
            # Check if we have scaffold content for this file
            scaffold_content = None
            for sf in scaffold_context:
                if sf.get("path") == new_path and sf.get("valid"):
                    scaffold_content = sf.get("content", "")
                    break
            
            if scaffold_content:
                # Use pre-validated scaffold content
                logger.info(f"Using scaffold content for new file: {new_path}")
                all_inferred.append({
                    "path": new_path,
                    "patched": scaffold_content,
                    "is_new_file": True,
                    "from_scaffold": True,
                })
                previous_changes.append({
                    "path": new_path,
                    "changes_made": [f"Created {new_file.get('purpose', 'new file')} (from scaffold)"],
                })
            else:
                # Generate content via LLM
                logger.info(f"Creating new file: {new_path}")
                new_file_prompt = self._build_new_file_prompt(new_file, plan)
                try:
                    # Log LLM call for new file
                    call_id = log_llm_call(
                        "refactor", "new_file",
                        new_file_prompt,
                        context={"new_file": new_path}
                    )
                    total_prompt_chars += len(new_file_prompt)
                    llm_call_count += 1
                    
                    file_start = datetime.now()
                    new_file_response = call_llm(self.llm, new_file_prompt)
                    file_duration = (datetime.now() - file_start).total_seconds() * 1000
                    
                    response_str = str(new_file_response)
                    log_llm_response(
                        "refactor", "new_file",
                        new_file_response,
                        call_id=call_id,
                        duration_ms=file_duration
                    )
                    total_response_chars += len(response_str)
                    
                    llm_responses.append(response_str)
                    new_content = self._parse_new_file_response(new_file_response)
                    if new_content:
                        all_inferred.append({
                            "path": new_path,
                            "patched": new_content,
                            "is_new_file": True,
                        })
                        previous_changes.append({
                            "path": new_path,
                            "changes_made": ["Created new file"],
                        })
                except Exception as e:
                    logger.warning(f"Failed to create new file {new_path}: {e}")
        
        # Process each existing file
        for file_path in execution_order:
            # Find the file content
            file_content = None
            for f in cycle_spec.files:
                if f.path == file_path or f.path.endswith(file_path.split('/')[-1].split('\\')[-1]):
                    file_content = f.content
                    file_path = f.path  # Use the full path
                    break
            
            if file_content is None:
                logger.warning(f"File not found in cycle spec: {file_path}")
                continue
            
            logger.info(f"Processing file: {file_path}")
            
            # Build per-file prompt
            file_prompt = self._build_per_file_prompt(
                file_path, file_content, plan, previous_changes
            )
            logger.debug(f"Per-file prompt: {len(file_prompt)} chars")
            
            try:
                # Log LLM call for this file
                call_id = log_llm_call(
                    "refactor", "file_patch",
                    file_prompt,
                    context={"file_path": file_path, "file_num": execution_order.index(file_path) + 1}
                )
                total_prompt_chars += len(file_prompt)
                llm_call_count += 1
                
                file_start = datetime.now()
                file_response = call_llm(self.llm, file_prompt)
                file_duration = (datetime.now() - file_start).total_seconds() * 1000
                
                response_str = str(file_response)
                log_llm_response(
                    "refactor", "file_patch",
                    file_response,
                    call_id=call_id,
                    duration_ms=file_duration
                )
                total_response_chars += len(response_str)
                
                llm_responses.append(response_str)
                logger.debug(f"File response: {len(response_str)} chars")
                
                # Parse the per-file response
                file_patches = self._parse_per_file_response(file_response, file_path)
                
                if file_patches:
                    all_inferred.append(file_patches)
                    previous_changes.append({
                        "path": file_path,
                        "changes_made": file_patches.get("changes_made", ["patches applied"]),
                    })
                    logger.info(f"  -> {len(file_patches.get('search_replace', []))} search/replace operations")
                else:
                    logger.warning(f"  -> No patches parsed for {file_path}")
                    
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue
        
        # Now process all inferred patches
        logger.info(f"Applying patches from {len(all_inferred)} file responses...")
        
        # Convert to the format expected by _process_single_file_patch
        files_dict = [{"path": f.path, "content": f.content} for f in cycle_spec.files]
        
        # Build inferred list in standard format
        normalized_inferred = []
        for inf in all_inferred:
            if inf.get("is_new_file"):
                # New file - add as full patched content
                normalized_inferred.append({
                    "path": inf["path"],
                    "patched": inf["patched"],
                })
            elif "search_replace" in inf or "append" in inf or "prepend" in inf:
                # Search/replace and/or append/prepend format
                entry = {"path": inf["path"]}
                if "search_replace" in inf:
                    entry["search_replace"] = inf["search_replace"]
                if "append" in inf:
                    entry["append"] = inf["append"]
                if "prepend" in inf:
                    entry["prepend"] = inf["prepend"]
                normalized_inferred.append(entry)
            elif "patched" in inf:
                normalized_inferred.append(inf)
        
        # Process each file
        processing_results: List[PatchProcessingResult] = []
        
        for f in cycle_spec.files:
            result = self._process_single_file_patch(
                path=f.path,
                original=f.content or "",
                inferred=normalized_inferred,
            )
            processing_results.append(result)
        
        # Handle new files
        for inf in all_inferred:
            if inf.get("is_new_file"):
                processing_results.append(PatchProcessingResult(
                    path=inf["path"],
                    original="",
                    patched=inf["patched"],
                    diff=f"New file created with {len(inf['patched'])} chars",
                    status="applied",
                    warnings=[],
                    confidence=1.0,
                    applied_blocks=1,
                    total_blocks=1,
                    revert_reason="",
                    pre_validated=False,
                    validation_issues=[],
                    has_critical_error=False,
                    original_patched=None,
                ))
        
        # Build final proposal (reuse existing logic for atomic handling)
        return self._build_proposal_from_results(
            processing_results, 
            "\n---\n".join(llm_responses),
            plan.get("strategy", "sequential_mode")
        )
    
    def _parse_plan_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse the planning phase LLM response."""
        try:
            text = response if isinstance(response, str) else str(response)
            # Find JSON in response
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning(f"Failed to parse plan response: {e}")
        return None
    
    def _parse_per_file_response(self, response: str, file_path: str) -> Optional[Dict[str, Any]]:
        """Parse the per-file execution response."""
        try:
            text = response if isinstance(response, str) else str(response)
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                data = json.loads(json_match.group())
                # Ensure path is set
                if "path" not in data:
                    data["path"] = file_path
                return data
        except Exception as e:
            logger.warning(f"Failed to parse per-file response: {e}")
        return None
    
    def _build_new_file_prompt(self, new_file_spec: Dict[str, Any], plan: Dict[str, Any]) -> str:
        """Build prompt for creating a new file."""
        return f"""Create the following new file as part of a refactoring.

## New File Details
- Path: {new_file_spec.get('path', 'unknown')}
- Purpose: {new_file_spec.get('purpose', 'unknown')}
- Description: {new_file_spec.get('content_description', 'No description')}

## Refactoring Plan Context
Strategy: {plan.get('strategy', 'unknown')}
Summary: {plan.get('summary', '')}

Output ONLY the complete file content, no JSON wrapper. Start directly with the code.
"""
    
    def _parse_new_file_response(self, response: str) -> Optional[str]:
        """Parse the new file creation response."""
        text = response if isinstance(response, str) else str(response)
        # Remove any markdown code blocks
        text = re.sub(r'^```\w*\n?', '', text.strip())
        text = re.sub(r'\n?```$', '', text.strip())
        return text if text else None
    
    def _build_proposal_from_results(
        self,
        processing_results: List[PatchProcessingResult],
        llm_response: str,
        strategy: str = "unknown",
    ) -> AgentResult:
        """Build AgentResult from processing results (shared by both modes)."""
        # Check atomic proposal settings
        atomic_proposal = self.refactor_config.atomic_proposal
        revert_on_critical = self.refactor_config.revert_on_any_critical
        
        has_any_critical = any(r.has_critical_error for r in processing_results)
        has_any_failed = any(r.status == "failed" for r in processing_results)
        
        revert_all = False
        revert_all_reason = ""
        
        if atomic_proposal:
            if has_any_critical and revert_on_critical:
                critical_files = [r.path for r in processing_results if r.has_critical_error]
                revert_all = True
                revert_all_reason = f"Atomic proposal mode: critical errors in {critical_files}"
                logger.error(f"Atomic proposal: reverting all {len(processing_results)} files due to critical errors in: {critical_files}")
            elif has_any_failed:
                failed_files = [r.path for r in processing_results if r.status == "failed"]
                revert_all = True
                revert_all_reason = f"Atomic proposal mode: failed files {failed_files}"
                logger.error(f"Atomic proposal: reverting all {len(processing_results)} files due to failures in: {failed_files}")
        
        # Build patches and reverted lists
        patches = []
        reverted_files = []
        
        for result in processing_results:
            if revert_all and result.status not in ("unchanged",):
                # Revert this file
                reverted_files.append(RevertedFile(
                    path=result.path,
                    reason=revert_all_reason,
                    warnings=result.warnings,
                    original_patched=result.patched if result.patched != result.original else None,
                ))
                patches.append(Patch(
                    path=result.path,
                    original=result.original,
                    patched=result.original,  # Revert to original
                    diff="",
                    status="reverted",
                    warnings=result.warnings,
                    confidence=result.confidence,
                    applied_blocks=0,
                    total_blocks=result.total_blocks,
                    revert_reason=revert_all_reason,
                ))
            else:
                patches.append(Patch(
                    path=result.path,
                    original=result.original,
                    patched=result.patched,
                    diff=result.diff,
                    status=result.status,
                    warnings=result.warnings,
                    confidence=result.confidence,
                    applied_blocks=result.applied_blocks,
                    total_blocks=result.total_blocks,
                    revert_reason=result.revert_reason,
                    pre_validated=result.pre_validated,
                    validation_issues=result.validation_issues,
                ))
        
        # Count results
        changed = len([p for p in patches if p.diff and p.status == "applied"])
        reverted = len(reverted_files)
        
        logger.info(f"RefactorAgent completed: {changed}/{len(patches)} files changed, {reverted} reverted (atomic={atomic_proposal})")
        
        # Note: Roadmap building is done in the calling method that has full context
        # (cycle_spec, scaffold_result, llm_responses, etc.)
        
        proposal = RefactorProposal(
            patches=patches,
            reverted_files=reverted_files,
            rationale=f"Sequential mode with strategy: {strategy}",
            llm_response=llm_response,
        )
        
        # Include roadmap in output if generated
        output = proposal.model_dump()
        
        return AgentResult(status="success", output=output)

    def _build_prompt(self, cycle: CycleSpec, description: CycleDescription) -> str:
        """Basic prompt builder (kept for compatibility)."""
        return self._build_prompt_with_strategy(cycle, description, None, None)

    def _build_prompt_with_strategy(
        self,
        cycle: CycleSpec,
        description: CycleDescription,
        feedback: Optional[ValidationReport] = None,
        strategy: Optional[str] = None,
    ) -> str:
        """Build prompt with strategy guidance, examples, and chain-of-thought.
        
        Uses context budget management to optimize for limited context window.
        Adapts budget based on feedback type (syntax vs semantic errors).
        """
        file_paths = cycle.get_file_paths()
        cycle_dict = cycle.model_dump()
        has_feedback = feedback is not None
        
        # Analyze feedback to determine issue types
        has_syntax_errors = False
        has_cycle_issues = False
        syntax_error_files = set()
        
        if feedback and feedback.issues:
            for issue in feedback.issues:
                issue_type = getattr(issue, 'issue_type', 'semantic')
                if issue_type == 'syntax':
                    has_syntax_errors = True
                    if issue.path:
                        syntax_error_files.add(issue.path)
                elif issue_type == 'cycle':
                    has_cycle_issues = True
        
        # Log the feedback classification
        if has_feedback:
            logger.info(f"Feedback analysis: syntax_errors={has_syntax_errors}, cycle_issues={has_cycle_issues}, syntax_files={len(syntax_error_files)}")
        
        # Create context budget - adjust based on feedback type
        # If only syntax errors, skip RAG and give more room to files
        skip_rag = has_syntax_errors and not has_cycle_issues
        
        budget = create_budget_for_agent(
            "refactor", 
            total_tokens=self.context_window,
            has_feedback=has_feedback
        )
        
        # If we have syntax errors, reallocate RAG budget to file content
        if skip_rag:
            rag_budget = budget.get_token_budget(BudgetCategory.RAG_CONTEXT)
            logger.info(f"Syntax errors detected - reallocating {rag_budget} RAG tokens to file content")
            # Zero out RAG and add to file content
            budget.allocations[BudgetCategory.RAG_CONTEXT] = 0
            budget.allocations[BudgetCategory.FILE_CONTENT] += rag_budget / self.context_window
        
        # Get validation issues for file prioritization (if retry)
        validation_issues = None
        if feedback and feedback.issues:
            validation_issues = [{"path": i.path, "comment": i.comment} for i in feedback.issues]
        
        # Prioritize files based on cycle structure, validation issues, and strategy
        files_data = [{"path": f.path, "content": f.content or ""} for f in cycle.files]
        file_priorities = prioritize_cycle_files(
            files_data,
            cycle_dict.get("graph", {}),
            validation_issues=validation_issues,
            total_char_budget=budget.get_char_budget(BudgetCategory.FILE_CONTENT),
            strategy_hint=strategy,
        )
        
        # If we have syntax errors, boost priority of files with errors
        if syntax_error_files:
            for fp in file_priorities:
                # Check if this file has syntax errors (match by basename)
                basename = fp.path.split('/')[-1].split('\\')[-1]
                for err_path in syntax_error_files:
                    err_basename = err_path.split('/')[-1].split('\\')[-1]
                    if basename == err_basename:
                        fp.priority_score = 1.0  # Maximum priority
                        fp.reasons = ["syntax_error_file"]  # Replace reasons list
                        logger.debug(f"Boosted priority for syntax error file: {fp.path}")
                        break
        
        # Build file snippets with priority-based budgets
        snippets = []
        total_file_tokens = 0
        file_token_budget = budget.get_token_budget(BudgetCategory.FILE_CONTENT)
        
        for f in cycle.files:
            if total_file_tokens >= file_token_budget:
                logger.debug(f"File budget exhausted, skipping remaining files")
                break
                
            content = f.content or ""
            file_budget = get_file_budget(f.path, file_priorities, self.max_file_chars)
            
            # Select snippet with file-specific budget
            snippet = select_relevant_snippet(content, f.path, cycle_dict, file_budget)
            snippet_tokens = TokenEstimator.estimate_tokens_for_file(snippet, f.path)
            
            # Check if we have room
            if total_file_tokens + snippet_tokens <= file_token_budget:
                snippets.append(f"--- FILE: {f.path} ---\n{snippet}")
                total_file_tokens += snippet_tokens
            else:
                # Truncate to fit remaining budget
                remaining_tokens = file_token_budget - total_file_tokens
                truncated = truncate_to_token_budget(snippet, remaining_tokens)
                snippets.append(f"--- FILE: {f.path} ---\n{truncated}")
                total_file_tokens = file_token_budget
                break

        file_snippets = "\n\n".join(snippets) if snippets else ""
        budget.use_budget(BudgetCategory.FILE_CONTENT, total_file_tokens)
        logger.debug(f"File content: {total_file_tokens} tokens used across {len(snippets)} files")
        
        # Get RAG context with budget limit (skip if we have syntax errors)
        rag_context = ""
        if skip_rag:
            logger.info("Skipping RAG context due to syntax errors - focusing on file content")
        else:
            rag_token_budget = budget.get_token_budget(BudgetCategory.RAG_CONTEXT)
            rag_context = self._get_rag_context(cycle, strategy, description)
            if rag_context:
                rag_context = truncate_to_token_budget(rag_context, rag_token_budget, "prose")
        
        # Get pattern example (skip if we have syntax errors - focus on fixing the code)
        # Also skip for compact prompts to save tokens
        pattern_example = ""
        if not skip_rag and not self.use_compact_prompts:
            pattern_example = self._get_pattern_example(strategy)

        # Use compact template if available and appropriate
        template_path = self._get_prompt_template()
        
        if template_path:
            tpl = load_template(template_path)
            result = safe_format(
                tpl,
                id=cycle.id,
                graph=json.dumps(cycle.graph.model_dump()),
                files=", ".join(file_paths),
                description=description.text,
                file_snippets=file_snippets,
                rag_context=rag_context,
                pattern_example=pattern_example,
                strategy=strategy or "not specified",
            )
            # Append snippets if template didn't include them
            contains_file_blocks = any((f"--- FILE: {p}" in result) for p in file_paths)
            if file_snippets and "{file_snippets}" not in tpl and not contains_file_blocks:
                result = result + "\n\n" + file_snippets
            
            # Append feedback if provided
            if feedback:
                result += self._format_feedback(feedback)
            
            # Validate and truncate if prompt exceeds context window
            result = self._enforce_prompt_budget(result, file_snippets, file_paths)
            return result

        # Default prompt with chain-of-thought structure
        prompt = f"""You are a refactoring expert. Break the cyclic dependency described below.

## Cycle Information
- ID: {cycle.id}
- Nodes: {', '.join(cycle.graph.nodes)}
- Edges: {json.dumps(cycle.graph.edges)}
- Files: {', '.join(file_paths)}

## Problem Description (from analysis)
{description.text[:2000]}

## Recommended Strategy: {strategy or "Choose the most appropriate"}

{pattern_example}

## Source Code
{file_snippets}
"""

        if rag_context:
            prompt += f"""
## Reference (from architecture literature)
{rag_context[:1500]}
"""

        if feedback:
            prompt += self._format_feedback(feedback)

        prompt += """
## IMPORTANT: Think Step-by-Step

Before writing code, reason through:
1. IDENTIFY: Which specific import/reference creates the problematic edge?
2. STRATEGY: Which pattern applies? (interface extraction, dependency inversion, shared module)
3. PLAN: What minimal changes break the cycle without breaking functionality?
4. IMPLEMENT: Write the patches

## Output Format - CHOOSE ONE:

### Option A: For SMALL files (< 100 lines) - Full content
```json
{
  "reasoning": "<brief explanation>",
  "strategy_used": "<interface_extraction|dependency_inversion|shared_module|other>",
  "patches": [
    {"path": "path/to/file.ext", "patched": "<COMPLETE file content>"}
  ]
}
```

### Option B: For LARGE files - Use SEARCH/REPLACE blocks (PREFERRED)
```json
{
  "reasoning": "<brief explanation>",
  "strategy_used": "<interface_extraction|dependency_inversion|shared_module|other>",
  "changes": [
    {
      "path": "path/to/file.ext",
      "search_replace": [
        {
          "search": "<exact text to find, 3-5 lines of context>",
          "replace": "<replacement text>"
        }
      ]
    }
  ]
}
```

### Option C: Use file markers with SEARCH/REPLACE
```
--- FILE: path/to/file.ext ---
<<<<<<< SEARCH
using OldNamespace;
=======
using NewNamespace;
>>>>>>> REPLACE

<<<<<<< SEARCH
public class Foo : IOldInterface
=======
public class Foo : INewInterface
>>>>>>> REPLACE
```

## Critical Requirements
1. For large files, use SEARCH/REPLACE blocks (Option B or C) - this prevents truncation
2. Include enough context in SEARCH blocks to be unique (3-5 lines before/after)
3. Ensure all brackets/braces are balanced in your output
4. Preserve existing functionality
5. DO NOT truncate your output - if file is too large, use SEARCH/REPLACE
"""
        # Validate and truncate if prompt exceeds context window
        prompt = self._enforce_prompt_budget(prompt, file_snippets, file_paths)
        return prompt
    
    def _enforce_prompt_budget(
        self, 
        prompt: str, 
        file_snippets: str,
        file_paths: List[str],
    ) -> str:
        """Ensure prompt fits within context window, truncating if necessary.
        
        Reserves ~30% of context for LLM output.
        If prompt exceeds budget, progressively truncates:
        1. RAG context
        2. Pattern examples
        3. File snippets (keeping most relevant)
        
        Args:
            prompt: The full prompt text
            file_snippets: Original file snippets (for potential re-truncation)
            file_paths: List of file paths in the cycle
            
        Returns:
            Prompt that fits within context budget
        """
        # Reserve 30% for output, use 70% for prompt
        max_prompt_tokens = int(self.context_window * 0.7)
        max_prompt_chars = max_prompt_tokens * 4  # ~4 chars per token
        
        current_len = len(prompt)
        
        if current_len <= max_prompt_chars:
            logger.debug(f"Prompt within budget: {current_len} chars <= {max_prompt_chars} max")
            return prompt
        
        logger.warning(f"Prompt exceeds budget: {current_len} chars > {max_prompt_chars} max ({max_prompt_tokens} tokens). Truncating...")
        
        # Strategy 1: Remove RAG context section
        rag_marker = "## Reference (from architecture literature)"
        if rag_marker in prompt:
            # Find and remove the RAG section
            rag_start = prompt.find(rag_marker)
            # Find the next section (## header)
            next_section = prompt.find("\n## ", rag_start + len(rag_marker))
            if next_section == -1:
                next_section = prompt.find("\n---", rag_start + len(rag_marker))
            if next_section != -1:
                prompt = prompt[:rag_start] + prompt[next_section:]
                logger.debug(f"Removed RAG context. New length: {len(prompt)} chars")
                if len(prompt) <= max_prompt_chars:
                    return prompt
        
        # Strategy 2: Remove pattern examples
        pattern_markers = ["**Interface Extraction Pattern**", "**Dependency Inversion Pattern**", 
                          "**Shared Module Extraction Pattern**", "**Common Refactoring Patterns:**"]
        for marker in pattern_markers:
            if marker in prompt:
                pattern_start = prompt.find(marker)
                # Find end of pattern section (next ## or ---)
                pattern_end = prompt.find("\n## ", pattern_start + len(marker))
                if pattern_end == -1:
                    pattern_end = prompt.find("\n---", pattern_start + len(marker))
                if pattern_end != -1:
                    prompt = prompt[:pattern_start] + prompt[pattern_end:]
                    logger.debug(f"Removed pattern example. New length: {len(prompt)} chars")
                    if len(prompt) <= max_prompt_chars:
                        return prompt
        
        # Strategy 3: Truncate file snippets more aggressively
        file_section_start = prompt.find("## Source Code")
        if file_section_start == -1:
            file_section_start = prompt.find("--- FILE:")
        
        if file_section_start != -1:
            # Find end of file section
            file_section_end = prompt.find("\n## ", file_section_start + 20)
            if file_section_end == -1:
                file_section_end = prompt.find("\n---\n", file_section_start + 20)
            
            if file_section_end != -1:
                # Calculate how much we need to cut
                excess = len(prompt) - max_prompt_chars
                file_section = prompt[file_section_start:file_section_end]
                
                # Truncate file section content
                if len(file_section) > excess + 500:  # Keep at least 500 chars
                    # Keep header and truncate content
                    header_end = file_section.find("\n", 20) + 1
                    header = file_section[:header_end] if header_end > 0 else "## Source Code\n"
                    available = len(file_section) - excess - 200
                    truncated_files = file_section[header_end:header_end + available]
                    truncated_files += f"\n\n... (truncated to fit context window - {len(file_paths)} files total) ..."
                    
                    prompt = prompt[:file_section_start] + header + truncated_files + prompt[file_section_end:]
                    logger.debug(f"Truncated file snippets. New length: {len(prompt)} chars")
        
        # Final fallback: hard truncate with warning
        if len(prompt) > max_prompt_chars:
            logger.error(f"Could not fit prompt in budget. Hard truncating from {len(prompt)} to {max_prompt_chars} chars")
            prompt = prompt[:max_prompt_chars - 100] + "\n\n... [TRUNCATED - context limit reached] ..."
        
        return prompt

    def _format_feedback(self, feedback: ValidationReport) -> str:
        """Format validator feedback for retry prompt.
        
        Categorizes issues by type and provides focused instructions.
        Includes enriched retry context when available.
        Detects anti-patterns and provides specific "do not" guidance.
        """
        # Access enriched retry context directly from model fields
        iteration = getattr(feedback, 'iteration', 1)
        remaining_attempts = getattr(feedback, 'remaining_attempts', 0)
        previous_reverted = getattr(feedback, 'previous_reverted_files', [])
        previous_summary = getattr(feedback, 'previous_attempt_summary', '')
        failed_strategies = getattr(feedback, 'failed_strategies', [])
        
        # Categorize issues
        syntax_issues = []
        cycle_issues = []
        semantic_issues = []
        
        for issue in feedback.issues:
            issue_type = getattr(issue, 'issue_type', 'semantic')
            if issue_type == 'syntax':
                syntax_issues.append(issue)
            elif issue_type == 'cycle':
                cycle_issues.append(issue)
            else:
                semantic_issues.append(issue)
        
        # Detect anti-patterns from failure history
        anti_patterns = self._detect_anti_patterns(
            syntax_issues, cycle_issues, semantic_issues, 
            previous_reverted, failed_strategies
        )
        
        result = f"\n\n## ⚠️ Attempt {iteration} Failed - {remaining_attempts} attempt(s) remaining\n"
        
        # Show anti-patterns as prominent warnings (highest priority)
        if anti_patterns:
            result += "\n### 🚫 DO NOT REPEAT THESE MISTAKES:\n"
            for ap in anti_patterns:
                result += f"- ❌ {ap}\n"
            result += "\n"
        
        # Show previous attempt summary if available
        if previous_summary:
            result += f"\n**Previous attempt:** {previous_summary}\n"
        
        # Show what was reverted (important context for LLM)
        if previous_reverted:
            result += "\n### 🔄 Files Reverted in Previous Attempt:\n"
            result += "These files had errors and were reverted to original. Try a DIFFERENT approach:\n"
            for rf in previous_reverted[:5]:
                path = rf.get('path', rf.path if hasattr(rf, 'path') else 'unknown')
                reason = rf.get('reason', rf.reason if hasattr(rf, 'reason') else 'unknown')
                result += f"- **{path}**: {reason[:100]}\n"
        
        # Show failed strategies so LLM tries something different
        if failed_strategies:
            result += f"\n**Strategies already tried:** {', '.join(failed_strategies)}\n"
            result += "Please try a DIFFERENT approach.\n"
        
        # Syntax errors first (highest priority)
        if syntax_issues:
            result += "\n### 🔴 CRITICAL - Syntax Errors (Fix First!):\n"
            result += "Your previous output had truncated or malformed code. Use SEARCH/REPLACE format for large files.\n\n"
            for issue in syntax_issues[:5]:  # Limit to avoid overwhelming
                line_info = f" (line {issue.line})" if issue.line else ""
                result += f"- **{issue.path}**{line_info}: {issue.comment}\n"
            if len(syntax_issues) > 5:
                result += f"- ... and {len(syntax_issues) - 5} more syntax issues\n"
            result += "\n**To fix:** Use targeted SEARCH/REPLACE blocks instead of outputting entire files.\n"
        
        # Cycle issues
        if cycle_issues:
            result += "\n### 🟠 Cycle Not Broken:\n"
            for issue in cycle_issues[:3]:
                result += f"- {issue.comment}\n"
            if len(cycle_issues) > 3:
                result += f"- ... and {len(cycle_issues) - 3} more cycle issues\n"
        
        # Other semantic issues
        if semantic_issues:
            result += "\n### 🟡 Other Issues:\n"
            for issue in semantic_issues[:5]:
                line_info = f" (line {issue.line})" if issue.line else ""
                result += f"- **{issue.path}**{line_info}: {issue.comment}\n"
        
        if feedback.suggestions:
            result += "\n**Suggestions:**\n"
            for suggestion in feedback.suggestions[:3]:
                result += f"- {suggestion}\n"
        
        result += "\nPlease fix these issues in your new proposal. Use a DIFFERENT strategy if previous attempts failed.\n"
        return result
    
    def _detect_anti_patterns(
        self,
        syntax_issues: List,
        cycle_issues: List,
        semantic_issues: List,
        previous_reverted: List[Dict],
        failed_strategies: List[str]
    ) -> List[str]:
        """Detect anti-patterns from failure history and return specific warnings.
        
        Analyzes the patterns in failures to provide actionable "do not" guidance.
        """
        anti_patterns = []
        
        # Pattern 1: Full file output causing syntax errors
        if syntax_issues:
            has_truncation = any(
                "truncat" in str(getattr(i, 'comment', '')).lower() or
                "incomplete" in str(getattr(i, 'comment', '')).lower() or
                "unexpected EOF" in str(getattr(i, 'comment', '')) or
                "unterminated" in str(getattr(i, 'comment', '')).lower()
                for i in syntax_issues
            )
            if has_truncation:
                anti_patterns.append(
                    "DO NOT output full file contents - the output gets truncated. "
                    "Use SEARCH/REPLACE blocks to make targeted changes."
                )
        
        # Pattern 2: Same files keep failing
        if previous_reverted:
            reverted_paths = {rf.get('path', '') for rf in previous_reverted if rf.get('path')}
            current_failed_paths = {
                getattr(i, 'path', '') 
                for i in (syntax_issues + semantic_issues) 
                if getattr(i, 'path', '')
            }
            repeated_failures = reverted_paths & current_failed_paths
            if repeated_failures:
                files = ', '.join(list(repeated_failures)[:3])
                anti_patterns.append(
                    f"DO NOT use the same approach for {files} - it failed before. "
                    "Try a different file or a different type of change."
                )
        
        # Pattern 3: Cycle not broken despite changes
        if cycle_issues and "refactoring approach (cycle not broken)" in failed_strategies:
            anti_patterns.append(
                "DO NOT move code without breaking the dependency. "
                "Consider: 1) Extract an interface, 2) Use dependency injection, "
                "3) Create a mediator/event system, 4) Move the shared code to a third module."
            )
        
        # Pattern 4: Multiple syntax errors suggests wrong output format
        if len(syntax_issues) >= 3:
            anti_patterns.append(
                "DO NOT output file contents directly - always use the SEARCH/REPLACE format: "
                "<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE"
            )
        
        # Pattern 5: Same edge keeps appearing in cycle issues
        cycle_edges = []
        for issue in cycle_issues:
            comment = str(getattr(issue, 'comment', ''))
            # Try to extract edge info like "A -> B still exists"
            if "->" in comment:
                cycle_edges.append(comment)
        if len(cycle_edges) >= 2:
            anti_patterns.append(
                "The same dependency edge keeps appearing. "
                "Focus on REMOVING or INVERTING this specific dependency, "
                "not just moving code around."
            )
        
        return anti_patterns
    
    # -------------------------------------------------------------------------
    # Patch Parsing Methods (delegated to utils.patch_parser)
    # -------------------------------------------------------------------------
    
    def _parse_json_patches(self, text: str) -> List[Dict[str, str]]:
        """Parse patches from JSON format. Delegates to utils.patch_parser."""
        return parse_json_patches(text)
    
    def _extract_patches_from_data(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract patches from parsed JSON data. Delegates to utils.patch_parser."""
        return extract_patches_from_data(data)
    
    def _clean_code_content(self, content: str) -> str:
        """Remove code block markers. Delegates to utils.patch_parser."""
        return clean_code_content(content)
    
    def _parse_marker_patches(self, text: str) -> List[Dict[str, str]]:
        """Parse file marker format. Delegates to utils.patch_parser."""
        return parse_marker_patches(text)
    
    def _parse_search_replace_json(self, text: str) -> List[Dict[str, Any]]:
        """Parse JSON format with search_replace changes. Delegates to utils.patch_parser."""
        return parse_search_replace_json(text)
    
    # -------------------------------------------------------------------------
    # Patch Application Methods (delegated to utils.patch_applier)
    # -------------------------------------------------------------------------
    
    def _extract_line_hint(self, text: str) -> Optional[int]:
        """Extract line number hint from text. Delegates to utils.patch_applier."""
        return extract_line_hint(text)
    
    def _try_find_search_text(
        self, 
        search_text: str, 
        content_lines: List[str],
        content_lines_stripped: List[str],
        line_hint: Optional[int] = None
    ) -> Tuple[Optional[int], Optional[int], str, float]:
        """Try to find search text in content. Delegates to utils.patch_applier."""
        result = try_find_search_text(search_text, content_lines, content_lines_stripped, line_hint)
        return result.start, result.end, result.strategy, result.confidence
    
    def _apply_search_replace_atomic(
        self, 
        original: str, 
        search_replace_blocks: str,
        min_confidence: Optional[float] = None
    ) -> SearchReplaceResult:
        """Apply SEARCH/REPLACE blocks atomically. Delegates to utils.patch_applier."""
        if min_confidence is None:
            min_confidence = self.refactor_config.min_match_confidence
        return apply_search_replace_atomic(
            original, 
            search_replace_blocks,
            min_confidence=min_confidence,
            warn_confidence=self.refactor_config.warn_confidence,
            allow_low_confidence=self.refactor_config.allow_low_confidence
        )
    
    def _apply_search_replace_list_atomic(
        self, 
        original: str, 
        search_replace_list: List[Dict[str, str]],
        min_confidence: Optional[float] = None
    ) -> SearchReplaceResult:
        """Apply a list of search/replace operations atomically. Delegates to utils.patch_applier."""
        if min_confidence is None:
            min_confidence = self.refactor_config.min_match_confidence
        return apply_search_replace_list_atomic(
            original, 
            search_replace_list,
            min_confidence=min_confidence,
            warn_confidence=self.refactor_config.warn_confidence,
            allow_low_confidence=self.refactor_config.allow_low_confidence
        )
    
    def _apply_with_partial_rollback(
        self,
        original: str,
        search_replace_blocks: str,
        path: str,
        original_content: str = None
    ) -> SearchReplaceResult:
        """Apply with partial rollback on failure. Delegates to utils.patch_applier."""
        return apply_with_partial_rollback(
            original,
            search_replace_blocks,
            path,
            original_content,
            min_confidence=self.refactor_config.min_match_confidence,
            warn_confidence=self.refactor_config.warn_confidence,
            allow_low_confidence=self.refactor_config.allow_low_confidence
        )
    
    def _apply_search_replace(self, original: str, search_replace_blocks: str) -> Tuple[str, List[str]]:
        """Apply SEARCH/REPLACE blocks (simple interface). Delegates to utils.patch_applier."""
        result = self._apply_search_replace_atomic(original, search_replace_blocks)
        return result.content, result.warnings
    
    def _apply_search_replace_list(self, original: str, search_replace_list: List[Dict[str, str]]) -> Tuple[str, List[str]]:
        """Apply a list of search/replace operations (simple interface). Delegates to utils.patch_applier."""
        result = self._apply_search_replace_list_atomic(original, search_replace_list)
        return result.content, result.warnings
    
    # -------------------------------------------------------------------------
    # Diff and Validation Methods (delegated to utils.diff_utils)
    # -------------------------------------------------------------------------
    
    def _looks_truncated(self, text: str) -> bool:
        """Check if text appears truncated. Delegates to utils.diff_utils."""
        return looks_truncated(text)
    
    def _get_common_indent(self, lines: List[str]) -> str:
        """Get common leading whitespace. Delegates to utils.diff_utils."""
        return diff_get_common_indent(lines)
    
    def _validate_patched_content(self, original: str, patched: str, path: str) -> List[str]:
        """Validate patched content. Delegates to utils.diff_utils."""
        return validate_patched_content(original, patched, path)

    def _infer_patches(self, llm_response: Any, cycle_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse patches from LLM response using multiple strategies.
        
        Delegates to utils.patch_parser.infer_patches.
        """
        return infer_patches(llm_response, cycle_files)

    def _validate_inferred_patches(
        self, 
        inferred: List[Dict[str, Any]], 
        strategy_used: str = ""
    ) -> None:
        """Validate inferred patches for common LLM issues.
        
        Checks for:
        1. No-op patches (search == replace)
        2. Interface extraction without creating interface
        3. Hallucinated type references
        
        Logs warnings but doesn't block - patches may still be applied.
        Respects hallucination_detection config setting.
        
        Args:
            inferred: List of inferred patch dictionaries
            strategy_used: The strategy the LLM claimed to use
        """
        # Skip if hallucination detection is disabled
        if not self.refactor_config.hallucination_detection:
            logger.debug("Hallucination detection disabled in config, skipping validation")
            return
        
        from utils.syntax_checker import (
            detect_noop_patch, 
            validate_interface_extraction,
            detect_hallucinated_types
        )
        
        # Check for no-op patches
        noop_count = 0
        for patch in inferred:
            sr_list = patch.get("search_replace", [])
            for sr in sr_list:
                search = sr.get("search", "")
                replace = sr.get("replace", "")
                if detect_noop_patch(search, replace):
                    noop_count += 1
                    logger.warning(
                        f"No-op patch detected in {patch.get('path', 'unknown')}: "
                        f"search == replace (first 50 chars: '{search[:50]}...')"
                    )
        
        if noop_count > 0:
            logger.warning(f"Found {noop_count} no-op patches that will have no effect")
        
        # Validate interface extraction strategy
        if strategy_used:
            is_valid, warnings = validate_interface_extraction(inferred, strategy_used)
            for warning in warnings:
                logger.error(f"Strategy validation failed: {warning}")
        
        # Check for hallucinated types
        all_sr = []
        for patch in inferred:
            all_sr.extend(patch.get("search_replace", []))
        
        hallucinated = detect_hallucinated_types(all_sr)
        if hallucinated:
            logger.error(
                f"Possible hallucinated interface types detected: {hallucinated}. "
                "These types are introduced but may not be defined anywhere. "
                "Check if new_files contains the interface definitions."
            )
            
            # Check if any new files define these interfaces
            new_file_contents = ""
            for patch in inferred:
                if patch.get("is_new_file"):
                    new_file_contents += patch.get("patched", "") + "\n"
            
            undefined = [h for h in hallucinated if h not in new_file_contents]
            if undefined:
                logger.error(
                    f"CRITICAL: Interface types {undefined} are used but never defined! "
                    "The LLM is hallucinating interface extraction without creating interfaces."
                )

    def _make_unified_diff(self, original: str, patched: str, path: str) -> str:
        """Generate a unified diff. Delegates to utils.diff_utils."""
        return make_unified_diff(original, patched, path)

    def _check_for_truncation(self, content: str, path: str) -> bool:
        """Check if content appears to be truncated. Delegates to utils.diff_utils."""
        return check_for_truncation(content, path)
    
    def _log_failed_patch_details(
        self,
        path: str,
        original: str,
        patched_entry: Optional[Dict[str, Any]],
        warnings: List[str],
        reason: str,
    ) -> None:
        """Log detailed information about a failed patch for debugging.
        
        This logs at DEBUG level to help diagnose why patches fail to apply.
        Information is logged in a structured way for later analysis.
        """
        logger.debug("=" * 60)
        logger.debug(f"FAILED PATCH DETAILS: {path}")
        logger.debug("=" * 60)
        logger.debug(f"Failure reason: {reason}")
        
        # Log original file info
        original_lines = original.split('\n') if original else []
        logger.debug(f"Original file: {len(original_lines)} lines, {len(original)} chars")
        
        # Log first/last lines of original for context
        if original_lines:
            logger.debug(f"Original first 3 lines:")
            for i, line in enumerate(original_lines[:3]):
                logger.debug(f"  {i+1}: {line[:100]}")
            if len(original_lines) > 6:
                logger.debug(f"  ... ({len(original_lines) - 6} lines omitted) ...")
            logger.debug(f"Original last 3 lines:")
            for i, line in enumerate(original_lines[-3:]):
                logger.debug(f"  {len(original_lines) - 2 + i}: {line[:100]}")
        
        # Log what the LLM tried to do
        if patched_entry:
            logger.debug("-" * 40)
            logger.debug("LLM patch entry:")
            
            if patched_entry.get("patched"):
                patched_content = patched_entry.get("patched", "")
                patched_lines = patched_content.split('\n')
                logger.debug(f"  Type: full patched content ({len(patched_lines)} lines)")
                logger.debug(f"  First 3 lines:")
                for line in patched_lines[:3]:
                    logger.debug(f"    {line[:100]}")
            
            elif patched_entry.get("search_replace"):
                sr_list = patched_entry.get("search_replace", [])
                logger.debug(f"  Type: JSON search_replace ({len(sr_list)} operations)")
                for i, sr in enumerate(sr_list[:5]):  # Log up to 5 operations
                    search_text = sr.get("search", "")
                    replace_text = sr.get("replace", "")
                    search_lines = search_text.split('\n')
                    replace_lines = replace_text.split('\n')
                    logger.debug(f"  Operation {i+1}:")
                    logger.debug(f"    Search ({len(search_lines)} lines):")
                    for line in search_lines[:3]:
                        logger.debug(f"      | {line[:80]}")
                    if len(search_lines) > 3:
                        logger.debug(f"      ... ({len(search_lines) - 3} more lines)")
                    logger.debug(f"    Replace ({len(replace_lines)} lines):")
                    for line in replace_lines[:3]:
                        logger.debug(f"      | {line[:80]}")
                    if len(replace_lines) > 3:
                        logger.debug(f"      ... ({len(replace_lines) - 3} more lines)")
                if len(sr_list) > 5:
                    logger.debug(f"  ... and {len(sr_list) - 5} more operations")
            
            elif patched_entry.get("search_replace_blocks"):
                blocks = patched_entry.get("search_replace_blocks", "")
                logger.debug(f"  Type: marker-style SEARCH/REPLACE ({len(blocks)} chars)")
                # Log first 500 chars of blocks
                logger.debug(f"  Content preview:")
                for line in blocks[:500].split('\n'):
                    logger.debug(f"    {line[:100]}")
                if len(blocks) > 500:
                    logger.debug(f"    ... ({len(blocks) - 500} more chars)")
        else:
            logger.debug("No patch entry found for this file")
        
        # Log warnings
        if warnings:
            logger.debug("-" * 40)
            logger.debug(f"Warnings ({len(warnings)}):")
            for w in warnings[:10]:
                logger.debug(f"  - {w}")
            if len(warnings) > 10:
                logger.debug(f"  ... and {len(warnings) - 10} more warnings")
        
        logger.debug("=" * 60)

    def _process_single_file_patch(
        self,
        path: str,
        original: str,
        inferred: List[Dict[str, Any]],
    ) -> PatchProcessingResult:
        """Process a single file's patch and return the result.
        
        This method handles:
        - Finding the matching patch from inferred list
        - Applying SEARCH/REPLACE operations
        - Pre-validating patched content for critical errors
        
        Args:
            path: File path being patched
            original: Original file content
            inferred: List of inferred patches from LLM
            
        Returns:
            PatchProcessingResult with all patch details
        """
        # Try multiple matching strategies to find the patch
        patched_entry = None
        basename = path.split("/")[-1]
        
        # 1. Exact path match
        patched_entry = next((p for p in inferred if p.get("path") == path), None)
        
        if patched_entry is None:
            # 2. Basename match
            patched_entry = next((p for p in inferred if p.get("path", "").endswith(basename)), None)
        
        if patched_entry is None:
            # 3. Partial path match
            for p in inferred:
                inferred_path = p.get("path", "")
                if inferred_path and (path.endswith(inferred_path) or inferred_path.endswith(basename)):
                    patched_entry = p
                    break
        
        if patched_entry is None:
            # 4. Case-insensitive basename match
            basename_lower = basename.lower()
            patched_entry = next(
                (p for p in inferred if p.get("path", "").lower().endswith(basename_lower)), 
                None
            )
        
        if patched_entry is None:
            logger.warning(f"No patch found for {path} (basename: {basename})")
        
        # Initialize tracking variables
        patch_status = "unchanged"
        patch_warnings: List[str] = []
        patch_confidence = 1.0
        applied_blocks = 0
        total_blocks = 0
        revert_reason = ""
        has_critical_error = False
        original_patched = None
        patched = original
        
        # Determine patched content
        if patched_entry:
            if patched_entry.get("patched"):
                # Full patched content provided
                patched = patched_entry.get("patched")
                patch_status = "applied"
            elif patched_entry.get("search_replace_blocks"):
                # Marker-style SEARCH/REPLACE blocks
                sr_result = self._apply_with_partial_rollback(
                    original, 
                    patched_entry.get("search_replace_blocks"),
                    path,
                    original
                )
                patched = sr_result.content
                patch_warnings = sr_result.warnings
                patch_confidence = sr_result.confidence
                applied_blocks = sr_result.applied_count
                total_blocks = sr_result.total_count
                
                if sr_result.applied_count == 0:
                    patch_status = "failed"
                    revert_reason = "No SEARCH/REPLACE blocks could be applied"
                    has_critical_error = True
                    # Log detailed failure info at DEBUG level
                    self._log_failed_patch_details(
                        path, original, patched_entry, sr_result.warnings, revert_reason
                    )
                elif not sr_result.is_atomic:
                    patch_status = "partial"
                else:
                    patch_status = "applied"
                
                logger.debug(f"Applied marker-style search/replace to {path}: {applied_blocks}/{total_blocks}")
            elif patched_entry.get("search_replace"):
                # JSON-style search_replace list
                sr_result = self._apply_search_replace_list_atomic(
                    original, patched_entry.get("search_replace")
                )
                patched = sr_result.content
                patch_warnings = sr_result.warnings
                patch_confidence = sr_result.confidence
                applied_blocks = sr_result.applied_count
                total_blocks = sr_result.total_count
                
                if sr_result.applied_count == 0:
                    patch_status = "failed"
                    revert_reason = "No search/replace operations could be applied"
                    has_critical_error = True
                    # Log detailed failure info at DEBUG level
                    self._log_failed_patch_details(
                        path, original, patched_entry, sr_result.warnings, revert_reason
                    )
                else:
                    patch_status = "applied"
                
                logger.debug(f"Applied JSON-style search/replace to {path}: {applied_blocks}/{total_blocks}")
            elif patched_entry.get("line_patches") or patched_entry.get("changes"):
                # Line-based patches - uses line numbers instead of text matching
                line_changes = patched_entry.get("line_patches") or patched_entry.get("changes", [])
                if not isinstance(line_changes, list):
                    line_changes = [line_changes]
                
                # Build patch structure expected by apply_line_patches
                patch_data = [{"path": path, "changes": line_changes}]
                
                line_result = apply_line_patches(original, patch_data)
                patched = line_result.content
                patch_warnings = line_result.warnings + line_result.errors
                patch_confidence = 1.0 if line_result.success else 0.5
                applied_blocks = line_result.applied_count
                total_blocks = line_result.total_count
                
                if line_result.applied_count == 0:
                    patch_status = "failed"
                    revert_reason = "No line patches could be applied"
                    has_critical_error = True
                    self._log_failed_patch_details(
                        path, original, patched_entry, patch_warnings, revert_reason
                    )
                else:
                    patch_status = "applied"
                
                logger.debug(f"Applied line-based patches to {path}: {applied_blocks}/{total_blocks}")
            
            # Handle append/prepend operations (can be combined with search_replace OR used alone)
            append_content = patched_entry.get("append")
            prepend_content = patched_entry.get("prepend")
            
            if append_content or prepend_content:
                from utils.patch_applier import apply_append_prepend
                new_content, append_warnings = apply_append_prepend(
                    patched, append=append_content, prepend=prepend_content
                )
                if new_content != patched:
                    patched = new_content
                    patch_warnings.extend(append_warnings)
                    # If this was an append-only patch (no search_replace), mark as applied
                    if patch_status == "unchanged":
                        patch_status = "applied"
                        patch_confidence = 1.0  # Append operations have full confidence
                    applied_blocks += 1
                    total_blocks += 1
                    logger.info(f"Applied append/prepend to {path}")
                elif patch_status == "unchanged" and (append_content or prepend_content):
                    # Append/prepend was requested but nothing changed - might be empty content
                    logger.warning(f"Append/prepend for {path} produced no changes")
        
        # Log warnings
        for warning in patch_warnings:
            logger.warning(f"S/R Warning [{path}]: {warning}")
        
        # Pre-validate patched content
        pre_validated = False
        validation_issues: List[str] = []
        repair_info = AutoRepairInfo()
        
        if patched != original:
            validation_result = validate_code_block(patched, path, original)
            pre_validated = True
            validation_issues = [i.message for i in validation_result.issues]
            
            for issue in validation_result.issues:
                if issue.severity == "critical":
                    logger.error(f"Patch Validation [{path}]: {issue.message}")
                else:
                    logger.warning(f"Patch Validation [{path}]: {issue.message}")
            
            # Attempt auto-repair if enabled and there are critical issues
            if (validation_result.has_critical and 
                self.refactor_config.auto_repair_syntax and
                should_attempt_repair(validation_result.issues)):
                
                logger.info(f"Attempting auto-repair for {path}...")
                repair_result = auto_repair_syntax(patched, path)
                repair_info = AutoRepairInfo(
                    attempted=True,
                    was_repaired=repair_result.was_repaired,
                    repairs_made=repair_result.repairs_made,
                    confidence=repair_result.confidence
                )
                
                if repair_result.was_repaired:
                    logger.info(f"Auto-repair successful for {path}: {repair_result}")
                    
                    # Check if repair meets confidence threshold
                    if repair_result.confidence >= self.refactor_config.auto_repair_min_confidence:
                        # Re-validate the repaired content
                        patched = repair_result.content
                        validation_result = validate_code_block(patched, path, original)
                        validation_issues = [i.message for i in validation_result.issues]
                        
                        # Add repair info to warnings
                        for repair in repair_result.repairs_made:
                            patch_warnings.append(f"Auto-repair: {repair}")
                        
                        logger.info(f"Auto-repair applied (confidence={repair_result.confidence:.0%}), "
                                   f"re-validation: {len(validation_result.issues)} issues remaining")
                    else:
                        logger.warning(f"Auto-repair confidence too low ({repair_result.confidence:.0%} < "
                                      f"{self.refactor_config.auto_repair_min_confidence:.0%}), not applying")
                else:
                    logger.debug(f"Auto-repair did not find fixable issues for {path}")
            
            # Check for critical issues (may have been fixed by auto-repair)
            if validation_result.has_critical:
                # Check config: should we block on validation failure?
                if self.refactor_config.block_on_validation_failure:
                    has_critical_error = True
                    revert_reason = "; ".join(i.message for i in validation_result.issues if i.severity == "critical")
                    original_patched = patched  # Save what we tried
                    # Log detailed failure info at DEBUG level
                    self._log_failed_patch_details(
                        path, original, patched_entry, patch_warnings + validation_issues, revert_reason
                    )
                else:
                    # Log but don't block
                    logger.warning(f"Critical validation issues in {path} but block_on_validation_failure=False, proceeding")
                    patch_warnings.append("Validation issues present but not blocking (block_on_validation_failure=False)")
        
        # Compile/lint check (if enabled and no critical errors yet)
        compile_info = CompileCheckInfo()
        
        if patched != original and self.refactor_config.compile_check and not has_critical_error:
            compile_result = self.compile_checker.check_file(path, patched)
            compile_info = CompileCheckInfo(
                checked=True,
                success=compile_result.success,
                tool_used=compile_result.tool_used,
                error_count=compile_result.error_count,
                warning_count=compile_result.warning_count,
                errors=[str(e) for e in compile_result.errors],
            )
            
            if compile_result.tool_available:
                logger.info(f"Compile check [{path}]: {compile_result.summary()}")
                
                if not compile_result.success:
                    for error in compile_result.errors:
                        logger.error(f"Compile Error [{path}]: {error}")
                        validation_issues.append(f"Compile: {error}")
                    
                    # Mark as critical if configured to revert on compile errors
                    if self.refactor_config.revert_on_compile_error:
                        has_critical_error = True
                        revert_reason = f"Compile check failed: {compile_result.error_count} errors"
                        original_patched = patched
                
                for warning in compile_result.warnings[:5]:  # Limit warnings
                    logger.warning(f"Compile Warning [{path}]: {warning}")
                    patch_warnings.append(f"Compile: {warning}")
            else:
                logger.debug(f"Compile check skipped for {path}: {compile_result.tool_used} not available")
        
        # Generate diff
        diff = self._make_unified_diff(original, patched, path) if patched != original else ""
        
        return PatchProcessingResult(
            path=path,
            original=original,
            patched=patched,
            diff=diff,
            status=patch_status,
            warnings=patch_warnings,
            confidence=patch_confidence,
            applied_blocks=applied_blocks,
            total_blocks=total_blocks,
            revert_reason=revert_reason,
            pre_validated=pre_validated,
            validation_issues=validation_issues,
            has_critical_error=has_critical_error,
            original_patched=original_patched,
            compile_info=compile_info,
            repair_info=repair_info,
        )

    def run(
        self,
        cycle_spec: Union[CycleSpec, Dict[str, Any]],
        description: Union[CycleDescription, Dict[str, Any]] = None,
        validator_feedback: Optional[Union[ValidationReport, Dict[str, Any]]] = None,
        prompt: str = None,
    ) -> AgentResult:
        """Propose patches to break the cycle using strategy-aware refactoring.

        Args:
            cycle_spec: CycleSpec model or dict with id, graph, files.
            description: CycleDescription model or dict from describer.
            validator_feedback: Optional ValidationReport for retry loop.
            prompt: Optional additional instructions.

        Returns:
            AgentResult with RefactorProposal output.
        """
        logger.info(f"RefactorAgent.run() starting, has_feedback={validator_feedback is not None}")

        # Convert inputs to models if needed
        if isinstance(cycle_spec, dict):
            cycle_spec = CycleSpec.model_validate(cycle_spec)
        if description is None:
            logger.error("Missing description input")
            return AgentResult(status="error", output=None, logs="Missing description")
        if isinstance(description, dict):
            description = CycleDescription.model_validate(description)
        if validator_feedback is not None and isinstance(validator_feedback, dict):
            validator_feedback = ValidationReport.model_validate(validator_feedback)
            logger.info(f"Retry with feedback: {len(validator_feedback.issues)} issues, {len(validator_feedback.suggestions)} suggestions")

        # =====================================================================
        # Mode Selection: Check for special modes before standard processing
        # =====================================================================
        
        # Suggestion mode - output suggestions for human review without applying patches
        # Only on first run (no validator feedback) since suggestions don't need retries
        if self.refactor_config.suggestion_mode and self.llm is not None and validator_feedback is None:
            logger.info("Using SUGGESTION MODE")
            return self._run_suggestion_mode(cycle_spec, description)
        
        # Line-based patching mode - uses line numbers instead of text matching
        # This mode DOES support retries with validator feedback
        if self.refactor_config.line_based_patching and self.llm is not None:
            if validator_feedback is not None:
                logger.info("Using LINE-BASED PATCH MODE (retry with feedback)")
            else:
                logger.info("Using LINE-BASED PATCH MODE")
            return self._run_line_patch_mode(cycle_spec, description, validator_feedback)
        
        # Minimal diff mode - focus on single smallest change
        if self.refactor_config.minimal_diff_mode and self.llm is not None and validator_feedback is None:
            logger.info("Using MINIMAL DIFF MODE")
            return self._run_minimal_diff_mode(cycle_spec, description)
        
        # Simple format mode - text-based output for smaller LLMs (7B-14B)
        # Can be explicitly enabled or auto-detected from model name
        use_simple = self.refactor_config.simple_format_mode
        if not use_simple and hasattr(self.llm, 'model'):
            # Auto-detect based on model name
            model_name = getattr(self.llm, 'model', '')
            use_simple = should_use_simple_format(
                model_name, 
                auto_threshold_b=self.refactor_config.auto_simple_threshold
            )
        
        if use_simple and self.llm is not None and validator_feedback is None:
            logger.info("Using SIMPLE FORMAT MODE")
            return self._run_simple_format_mode(cycle_spec, description)
        
        # Check if we should use sequential file mode
        num_files = len(cycle_spec.files)
        use_sequential = self._should_use_sequential_mode(num_files)
        
        if use_sequential and self.llm is not None and validator_feedback is None:
            # Use sequential mode for initial attempts (not retries with feedback)
            logger.info(f"Using sequential file mode for {num_files} files")
            return self._run_sequential_mode(cycle_spec, description, validator_feedback)
        elif use_sequential and validator_feedback is not None:
            logger.info("Sequential mode disabled for retry (using standard mode with feedback)")

        # Extract strategy hint from description
        strategy = self._extract_strategy_from_description(description)
        if strategy:
            logger.info(f"Using strategy: {strategy}")
        else:
            logger.info("No specific strategy detected, will use general approach")

        # Build strategy-aware prompt
        prompt_text = self._build_prompt_with_strategy(
            cycle_spec, description, validator_feedback, strategy
        )
        logger.debug(f"Built refactor prompt with {len(prompt_text)} chars")

        # If no LLM available just return original files as no-op patches
        if self.llm is None:
            logger.info("No LLM provided, returning no-op patches (original files unchanged)")
            patches = [
                Patch(path=f.path, original=f.content, patched=f.content, diff="")
                for f in cycle_spec.files
            ]
            proposal = RefactorProposal(patches=patches, rationale="No-op (no LLM provided)", llm_response=None)
            return AgentResult(status="success", output=proposal.model_dump())

        try:
            logger.info("Calling LLM for refactor proposal")
            
            # Log LLM input
            call_id = log_llm_call(
                "refactor", "main",
                prompt_text,
                context={
                    "cycle_id": cycle_spec.id,
                    "num_files": len(cycle_spec.files),
                    "strategy": strategy,
                    "has_feedback": validator_feedback is not None
                }
            )
            
            llm_start = datetime.now()
            llm_response = call_llm(self.llm, prompt_text)
            llm_duration = (datetime.now() - llm_start).total_seconds() * 1000
            
            # Log LLM output
            response_str = str(llm_response) if isinstance(llm_response, str) else json.dumps(llm_response)
            log_llm_response(
                "refactor", "main",
                llm_response,
                call_id=call_id,
                duration_ms=llm_duration
            )
            
            logger.debug(f"LLM response received, length: {len(response_str)} chars")

            # Convert files to dict for _infer_patches compatibility
            files_dict = [{"path": f.path, "content": f.content} for f in cycle_spec.files]
            inferred = self._infer_patches(llm_response, files_dict)
            logger.debug(f"Inferred {len(inferred)} patches from LLM response")

            # Try to extract strategy from response
            strategy_used = ""
            try:
                text = llm_response if isinstance(llm_response, str) else json.dumps(llm_response)
                json_match = re.search(r'\{[\s\S]*\}', text)
                if json_match:
                    parsed = json.loads(json_match.group())
                    if "strategy_used" in parsed:
                        strategy_used = parsed['strategy_used']
                        logger.info(f"LLM used strategy: {strategy_used}")
                    if "reasoning" in parsed:
                        logger.debug(f"LLM reasoning: {parsed['reasoning'][:200]}...")
            except Exception:
                pass

            # Validate the inferred patches before applying
            self._validate_inferred_patches(inferred, strategy_used)

            # Build final patches list merging originals with patched content
            # Log what patches were inferred for debugging
            if inferred:
                inferred_paths = [p.get("path", "unknown") for p in inferred]
                logger.info(f"Inferred patches for paths: {inferred_paths}")
            else:
                logger.warning("No patches inferred from LLM response - check if LLM returned valid output")
            
            # Phase 1: Process all files and collect results (without committing)
            processing_results: List[PatchProcessingResult] = []
            
            for f in cycle_spec.files:
                result = self._process_single_file_patch(
                    path=f.path,
                    original=f.content or "",
                    inferred=inferred,
                )
                processing_results.append(result)
            
            # Handle new files (interface extraction, shared modules, etc.)
            for inf in inferred:
                if inf.get("is_new_file") and inf.get("patched"):
                    new_path = inf.get("path", "")
                    new_content = inf.get("patched", "")
                    if new_path and new_content:
                        logger.info(f"Adding new file: {new_path}")
                        processing_results.append(PatchProcessingResult(
                            path=new_path,
                            original="",
                            patched=new_content,
                            diff=f"New file created with {len(new_content)} chars",
                            status="applied",
                            warnings=[],
                            confidence=1.0,
                            applied_blocks=1,
                            total_blocks=1,
                            revert_reason="",
                            pre_validated=False,
                            validation_issues=[],
                            has_critical_error=False,
                            original_patched=None,
                        ))
            
            # Phase 2: Check if atomic proposal mode should revert everything
            atomic_proposal = self.refactor_config.atomic_proposal
            revert_on_critical = self.refactor_config.revert_on_any_critical
            
            has_any_critical = any(r.has_critical_error for r in processing_results)
            has_any_failed = any(r.status == "failed" for r in processing_results)
            
            revert_all = False
            revert_all_reason = ""
            
            if atomic_proposal:
                if has_any_critical and revert_on_critical:
                    critical_files = [r.path for r in processing_results if r.has_critical_error]
                    revert_all = True
                    revert_all_reason = f"Atomic proposal mode: critical errors in {critical_files}"
                    logger.error(f"Atomic proposal: reverting all {len(processing_results)} files due to critical errors in: {critical_files}")
                elif has_any_failed:
                    failed_files = [r.path for r in processing_results if r.status == "failed"]
                    revert_all = True
                    revert_all_reason = f"Atomic proposal mode: patch application failed in {failed_files}"
                    logger.error(f"Atomic proposal: reverting all {len(processing_results)} files due to failures in: {failed_files}")
            
            # Phase 3: Build final patches list
            patches_out = []
            reverted_files = []
            files_changed = 0
            
            for result in processing_results:
                if revert_all:
                    # Revert everything - use original content
                    if result.patched != result.original:
                        reverted_files.append(RevertedFile(
                            path=result.path,
                            reason=revert_all_reason,
                            warnings=result.warnings,
                            original_patched=result.patched[:2000] if result.patched else None,
                        ))
                    
                    patches_out.append(Patch(
                        path=result.path,
                        original=result.original,
                        patched=result.original,  # Reverted to original
                        diff="",
                        status="reverted" if result.patched != result.original else "unchanged",
                        warnings=result.warnings + [revert_all_reason] if result.warnings else [revert_all_reason],
                        confidence=result.confidence,
                        applied_blocks=0,
                        total_blocks=result.total_blocks,
                        revert_reason=revert_all_reason if result.patched != result.original else "",
                        pre_validated=result.pre_validated,
                        validation_issues=result.validation_issues,
                    ))
                elif result.has_critical_error:
                    # Individual file revert (non-atomic mode)
                    logger.error(f"Reverting {result.path} due to critical errors")
                    reverted_files.append(RevertedFile(
                        path=result.path,
                        reason=result.revert_reason,
                        warnings=result.warnings,
                        original_patched=result.original_patched[:2000] if result.original_patched else None,
                    ))
                    
                    patches_out.append(Patch(
                        path=result.path,
                        original=result.original,
                        patched=result.original,  # Reverted
                        diff="",
                        status="reverted",
                        warnings=result.warnings,
                        confidence=result.confidence,
                        applied_blocks=0,
                        total_blocks=result.total_blocks,
                        revert_reason=result.revert_reason,
                        pre_validated=result.pre_validated,
                        validation_issues=result.validation_issues,
                    ))
                else:
                    # Normal case - apply the patch
                    if result.diff:
                        files_changed += 1
                        logger.debug(f"File changed: {result.path} ({len(result.diff)} chars diff)")
                    
                    patches_out.append(Patch(
                        path=result.path,
                        original=result.original,
                        patched=result.patched,
                        diff=result.diff,
                        status=result.status,
                        warnings=result.warnings,
                        confidence=result.confidence,
                        applied_blocks=result.applied_blocks,
                        total_blocks=result.total_blocks,
                        revert_reason=result.revert_reason,
                        pre_validated=result.pre_validated,
                        validation_issues=result.validation_issues,
                    ))

            logger.info(f"RefactorAgent completed: {files_changed}/{len(patches_out)} files changed, {len(reverted_files)} reverted (atomic={atomic_proposal})")
            
            # Build roadmap if in roadmap mode
            roadmap = None
            if self.refactor_config.roadmap_mode:
                # Convert patch results to dict format for roadmap builder
                patch_results = []
                for p in patches_out:
                    patch_results.append({
                        "path": p.path,
                        "original": p.original,
                        "patched": p.patched,
                        "diff": p.diff,
                        "status": p.status,
                        "warnings": p.warnings,
                        "confidence": p.confidence,
                        "revert_reason": p.revert_reason,
                        "validation_issues": p.validation_issues,
                    })
                
                roadmap = self._build_roadmap_output(
                    cycle_spec=cycle_spec,
                    strategy=strategy or "auto",
                    patch_results=patch_results,
                    llm_responses=[llm_response if isinstance(llm_response, str) else json.dumps(llm_response)],
                )
                logger.info(f"Roadmap generated: {len(roadmap.successful_patches)} success, "
                           f"{len(roadmap.partial_attempts)} partial, "
                           f"{len(roadmap.remaining_work)} remaining")
            
            proposal = RefactorProposal(
                patches=patches_out,
                reverted_files=reverted_files,
                rationale=f"LLM proposal using strategy: {strategy or 'auto-selected'}",
                llm_response=llm_response if isinstance(llm_response, str) else json.dumps(llm_response),
            )
            
            # Include roadmap in output if generated
            output = proposal.model_dump()
            if roadmap:
                output["roadmap"] = roadmap.model_dump()
                output["executive_summary"] = roadmap.executive_summary
            
            return AgentResult(status="success", output=output)
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            return AgentResult(status="error", output=None, logs=f"LLM call failed: {e}")
